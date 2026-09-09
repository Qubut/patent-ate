"""Machine-local 30k mixed-punctuation C-value timing gate."""

from __future__ import annotations

import time
from pathlib import Path

import ibis
import pyarrow as pa
import pytest

from patent_ate.cvalue.algebra import cvalue_expression, parent_windows
from patent_ate.plan import PATENT_ID_COLUMN

_TERM_STRUCT = pa.struct([
    ('term', pa.string()),
    ('frequency', pa.int64()),
    ('surfaces', pa.list_(pa.string())),
])
_COMPACT_SCHEMA = pa.schema([
    (PATENT_ID_COLUMN, pa.string()),
    ('n_docs', pa.int64()),
    ('terms', pa.list_(_TERM_STRUCT)),
])
_UNIQUE_TERMS = 30_000
_BUDGET_S = 0.4


@pytest.mark.benchmark
def test_cvalue_scores_30k_mixed_terms_under_budget(tmp_path: Path) -> None:
    groups = _UNIQUE_TERMS // 4
    terms = [
        term
        for index in range(groups)
        for term in (
            f'unit{index:04d} extra word',
            f'pre-unit{index:04d} extra word',
            f'foo{index:04d}/bar extra',
            f'a{index:04d}+b extra word',
        )
    ]
    assert len(set(terms)) == _UNIQUE_TERMS
    rows = [
        {
            PATENT_ID_COLUMN: 'p0',
            'n_docs': 1,
            'terms': [{'term': term, 'frequency': 1, 'surfaces': [term]} for term in terms],
        }
    ]
    table = pa.Table.from_pylist(rows, schema=_COMPACT_SCHEMA)
    spill = tmp_path / 'duckdb_tmp'
    spill.mkdir()
    backend = ibis.duckdb.connect(
        memory_limit='2GB',
        temp_directory=str(spill),
        threads=4,
        preserve_insertion_order=False,
    )
    try:
        patents = backend.create_table('patents', table)
        scored = cvalue_expression(patents)
        scored.to_polars()
        started = time.perf_counter()
        frame = scored.to_polars()
        elapsed = time.perf_counter() - started
        windows = int(
            parent_windows(
                patents
                .unnest('terms')
                .unpack('terms')
                .group_by('term')
                .agg(tf=ibis._.frequency.sum())
            )
            .count()
            .to_pyarrow()
            .as_py()
        )
        spill_bytes = sum(path.stat().st_size for path in spill.rglob('*') if path.is_file())
        cursor = backend.raw_sql(f'EXPLAIN {scored.compile()}')
        plan = '\n'.join(str(row[-1]) for row in cursor.fetchall()).upper()
    finally:
        backend.disconnect()
    print(
        f'elapsed_s={elapsed:.6f} windows={windows} spill_bytes={spill_bytes} height={frame.height}'
    )
    assert 'HASH_JOIN' in plan
    assert 'NESTED_LOOP_JOIN' not in plan
    assert 'CROSS_PRODUCT' not in plan
    assert frame.height > 0
    assert windows <= _UNIQUE_TERMS * 20
    assert spill_bytes == 0
    assert elapsed < _BUDGET_S, f'{elapsed:.4f}s over {_BUDGET_S}s budget'

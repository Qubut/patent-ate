"""Ray Data extract plan, compact rows, and DuckDB C-value."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import ibis
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray
from hypothesis import given, settings
from hypothesis import strategies as st
from jate import CValue
from jate.features import TermFrequency
from jate.models import Candidate, Document
from pydantic import ValidationError
from ray.data import ActorPoolStrategy

from patent_ate.corpus import sample_json_paths
from patent_ate.cvalue import score_term_parquet
from patent_ate.cvalue.algebra import (
    candidate_span_lengths,
    cvalue_expression,
    cvalue_keys,
    parent_contributions,
    parent_windows,
    scored_terms,
    surfaces,
    term_stats,
    termhood_table,
)
from patent_ate.cvalue.text import (
    JATE_CHUNK_PATTERN,
    JATE_WORD_CLASS,
    jate_word_count,
    normalized_key,
)
from patent_ate.extract import (
    EXTRACT_DIRNAME,
    PatentTermExtractor,
    configure_extract_context,
    corpus_termhood,
    extract_checkpoint_config,
    extract_ids,
    patent_rows,
    read_patent_plan,
    resolved_extract_workers,
    run_patent_extract,
)
from patent_ate.nlp import JateDraw
from patent_ate.plan import PATENT_ID_COLUMN, PLAN_FILENAME, ExtractPlan, PatentPlanRow
from patent_ate.ray_local import ensure_local_ray
from patent_ate.spec import AteSpec
from patent_ate.termhood import TermhoodStore, normalize_surface

_JSON_FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'hupd'
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
_PHRASES = (
    'coil spring',
    'spring assembly',
    'coil spring assembly',
    'vehicle interior',
    'foo_bar extra',
    "don't stop",
    'iso9001 extra',
    '-lock brake sensor',
    '(hydroxy)alkyl acrylate monomer',
    'nfκb pathway',
    '\u03b1helix structure',
)
_WINDOW_EQ_TERM = re.compile(
    r'ON\s+(?:"[^"]+"\.)?"window"\s*=\s*(?:"[^"]+"\.)?"term"',
    re.IGNORECASE,
)
_JOIN_EQ_PREDICATE = re.compile(
    r'ON\s+((?:"[^"]+"\.)?"[^"]+"\s*=\s*(?:"[^"]+"\.)?"[^"]+")',
    re.IGNORECASE,
)


def _write_compact(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_COMPACT_SCHEMA), path)
    return path


def _store_facts(path: Path) -> tuple[TermhoodStore, dict[str, float], dict[str, int]]:
    store = TermhoodStore.open(path)
    frame = pl.scan_parquet(store.parquet).collect()
    return (
        store,
        dict(frame.select('key', 'c_value').iter_rows()),
        dict(frame.select('key', 'df').iter_rows()),
    )


def _jate_table(rows: list[dict[str, Any]]) -> dict[str, float]:
    pool: dict[str, Candidate] = {}
    for row in rows:
        for item in row['terms']:
            term = str(item['term'])
            cand = pool.setdefault(
                term,
                Candidate(surface_form=term, normalized_form=term),
            )
            cand.surface_forms.update(item['surfaces'])
            for index in range(int(item['frequency'])):
                cand.add_position(str(row[PATENT_ID_COLUMN]), index, index + 1, 0)
    candidates = list(pool.values())
    docs = sum(int(row['n_docs']) for row in rows)
    freq = TermFrequency.build(candidates, docs)
    ranked = CValue().score(candidates, freq).filter_by_frequency(1).filter_by_length(min_words=2)
    return JateDraw.table_from_cvalue(ranked, freq).c_values


def test_extract_plan_writes_one_unique_id_per_patent(tmp_path: Path) -> None:
    paths = tuple(tmp_path / f'{index}.json' for index in range(8))
    for path in paths:
        path.write_text('{}', encoding='utf-8')
    plan = ExtractPlan.from_paths(paths)
    written = plan.write_parquet(tmp_path / 'plan.parquet')
    assert written == tmp_path / 'plan.parquet'
    assert written.is_file()
    assert not (tmp_path / 'plan').exists()
    assert len(plan.rows) == 8
    assert len(plan.ids) == 8
    assert all(row.patent_id == row.path for row in plan.rows)
    assert pq.read_table(written).num_rows == 8


def test_extract_blocks_uses_row_grain_not_workers() -> None:
    ate = AteSpec()
    assert ate.extract_block_rows == 256
    assert ate.extract_blocks(4_518_254) == 17_650
    assert ate.extract_blocks(100_000) == 391
    assert ate.extract_blocks(100) == 1
    assert ate.extract_blocks(0) == 0
    assert ate.extract_blocks(-3) == 0


def test_resolved_extract_workers_zero_uses_cpus() -> None:
    assert resolved_extract_workers(0) >= 1
    assert resolved_extract_workers(2) == 2


def test_extract_block_rows_must_cover_cpu_job_width() -> None:
    with pytest.raises(ValidationError, match='extract_block_rows must be at least cpu_job_width'):
        AteSpec(extract_block_rows=4, cpu_job_width=8)


def test_run_patent_extract_uses_extract_blocks_not_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Dataset:
        def map_batches(self, *_args: object, **_kwargs: object) -> _Dataset:
            captured['mapped'] = True
            return self

        def write_parquet(self, *_args: object, **_kwargs: object) -> None:
            captured['wrote'] = True

    def fake_read(table: object, **kwargs: object) -> _Dataset:
        captured['rows'] = getattr(table, 'num_rows', None)
        captured['kwargs'] = kwargs
        return _Dataset()

    monkeypatch.setattr('patent_ate.extract.ray.data.from_arrow', fake_read)
    plan = ExtractPlan(
        rows=tuple(PatentPlanRow(patent_id=f'p{index}', path=f'/p{index}') for index in range(64))
    ).write_parquet(tmp_path / 'plan.parquet')
    ate = AteSpec(extract_block_rows=8, cpu_job_width=8)
    run_patent_extract(
        plan,
        tmp_path / 'extract',
        tmp_path / 'ray_checkpoint',
        ate=ate,
        workers=3,
        batch_size=8,
    )
    assert captured['rows'] == 64
    kwargs = captured['kwargs']
    assert isinstance(kwargs, dict)
    assert ate.extract_blocks(64) == 8
    assert kwargs['override_num_blocks'] == 8
    assert kwargs['override_num_blocks'] != 3
    assert captured['mapped'] is True
    assert captured['wrote'] is True


def test_run_patent_extract_tail_uses_one_block_and_one_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Dataset:
        def map_batches(self, *_args: object, **_kwargs: object) -> _Dataset:
            captured['mapped'] = True
            return self

        def write_parquet(self, *_args: object, **_kwargs: object) -> None:
            captured['wrote'] = True

    def fake_read(table: object, **kwargs: object) -> _Dataset:
        captured['rows'] = getattr(table, 'num_rows', None)
        captured['ids'] = [str(value) for value in table.column(PATENT_ID_COLUMN).to_pylist()]
        captured['kwargs'] = kwargs
        return _Dataset()

    def capture_pool(**kwargs: object) -> ActorPoolStrategy:
        size = kwargs['size']
        captured['pool_size'] = size
        assert isinstance(size, int)
        return ActorPoolStrategy(size=size)

    monkeypatch.setattr('patent_ate.extract.ray.data.from_arrow', fake_read)
    monkeypatch.setattr('patent_ate.extract.ActorPoolStrategy', capture_pool)
    plan = ExtractPlan(
        rows=tuple(
            PatentPlanRow(patent_id=patent_id, path=f'/{patent_id}')
            for patent_id in ('p0', 'p1', 'p2')
        )
    ).write_parquet(tmp_path / 'plan.parquet')
    extract_dir = tmp_path / 'extract'
    _write_compact(
        extract_dir / 'seed.parquet',
        [
            {
                PATENT_ID_COLUMN: 'p0',
                'n_docs': 1,
                'terms': [{'term': 'seeded-only', 'frequency': 1, 'surfaces': ['seeded-only']}],
            }
        ],
    )
    run_patent_extract(
        plan,
        extract_dir,
        tmp_path / 'ray_checkpoint',
        ate=AteSpec(),
        workers=3,
        batch_size=8,
    )
    kwargs = captured['kwargs']
    assert isinstance(kwargs, dict)
    assert captured['rows'] == 2
    assert captured['ids'] == ['p1', 'p2']
    assert kwargs['override_num_blocks'] == 1
    assert captured['pool_size'] == 1
    assert captured['mapped'] is True
    assert captured['wrote'] is True
    assert not hasattr(_Dataset, 'count')


def _pass_through_map_stats(
    plan_path: Path,
    ate: AteSpec,
    *,
    batch_size: int,
    actors: int,
    out_dir: Path,
) -> tuple[tuple[int, ...], list[Path]]:
    class _PassThrough:
        def __init__(self) -> None:
            return

        def __call__(self, batch: pa.Table) -> pa.Table:
            return batch

    loaded = read_patent_plan(plan_path, ate=ate)
    assert loaded is not None
    mapped = loaded.dataset.map_batches(
        _PassThrough,
        batch_size=batch_size,
        batch_format='pyarrow',
        compute=ActorPoolStrategy(size=actors),
        num_cpus=1,
        num_gpus=0,
    )
    mapped.write_parquet(str(out_dir))
    executed = tuple(
        int(count)
        for count in re.findall(r'MapBatches[\s\S]{0,120}?(\d+) tasks executed', mapped.stats())
    )
    return executed, sorted(out_dir.glob('*.parquet'))


def test_read_patent_plan_runs_four_map_tasks_and_four_parts(tmp_path: Path) -> None:
    ate = AteSpec(extract_block_rows=8, cpu_job_width=8)
    ids = [f'p{index}' for index in range(32)]
    single = ExtractPlan(
        rows=tuple(PatentPlanRow(patent_id=patent_id, path=patent_id) for patent_id in ids)
    ).write_parquet(tmp_path / 'plan.parquet')
    out_dir = tmp_path / 'mapped'
    assert ate.extract_blocks(len(ids)) == 4

    owns_ray = False
    if ray.is_initialized():
        ray.shutdown()
    owns_ray = ensure_local_ray()
    ray.data.DataContext.get_current().checkpoint_config = None
    try:
        loaded = read_patent_plan(single, ate=ate)
        assert loaded is not None
        assert loaded.n_blocks == 4
        assert loaded.dataset.num_blocks() == 4
        executed, written_parts = _pass_through_map_stats(
            single, ate, batch_size=8, actors=2, out_dir=out_dir
        )
        assert executed
        assert max(executed) == 4
        assert len(written_parts) == 4
        written = [row for part in written_parts for row in pq.read_table(part).to_pylist()]
        assert len(written) == len(ids)
    finally:
        if owns_ray and ray.is_initialized():
            ray.shutdown()


def test_map_batches_bundles_blocks_smaller_than_batch_size(tmp_path: Path) -> None:
    ate = AteSpec(extract_block_rows=2, cpu_job_width=2)
    ids = [f'p{index}' for index in range(32)]
    plan = ExtractPlan(
        rows=tuple(PatentPlanRow(patent_id=patent_id, path=patent_id) for patent_id in ids)
    ).write_parquet(tmp_path / 'plan.parquet')
    assert ate.extract_blocks(32) == 16

    if ray.is_initialized():
        ray.shutdown()
    ensure_local_ray()
    ray.data.DataContext.get_current().checkpoint_config = None
    try:
        executed, written_parts = _pass_through_map_stats(
            plan, ate, batch_size=8, actors=2, out_dir=tmp_path / 'bundled'
        )
        assert executed
        assert max(executed) < 16
        written = [row for part in written_parts for row in pq.read_table(part).to_pylist()]
        assert len(written) == 32
    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_run_patent_extract_skips_completed_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_ids = ('p0', 'p1', 'p2')
    plan = ExtractPlan(
        rows=tuple(
            PatentPlanRow(patent_id=patent_id, path=f'/{patent_id}') for patent_id in plan_ids
        )
    ).write_parquet(tmp_path / 'plan.parquet')
    extract_dir = tmp_path / 'extract'
    seed_terms = [{'term': 'seeded-only', 'frequency': 1, 'surfaces': ['seeded-only']}]
    _write_compact(
        extract_dir / 'seed.parquet',
        [{PATENT_ID_COLUMN: 'p0', 'n_docs': 1, 'terms': seed_terms}],
    )
    submitted: list[list[str]] = []
    real_from_arrow = ray.data.from_arrow

    def capture_plan(table: pa.Table, **kwargs: object) -> object:
        submitted.append([str(value) for value in table.column(PATENT_ID_COLUMN).to_pylist()])
        return real_from_arrow(table, **kwargs)

    class _Recorder:
        def __init__(self, spacy_model: str, pipe_docs: int, sentence_group: int) -> None:
            return

        def __call__(self, batch: pa.Table) -> pa.Table:
            ids = [str(value) for value in batch.column(PATENT_ID_COLUMN).to_pylist()]
            return pa.Table.from_pylist(
                [
                    {
                        PATENT_ID_COLUMN: patent_id,
                        'n_docs': 1,
                        'terms': [{'term': 'fresh', 'frequency': 1, 'surfaces': ['fresh']}],
                    }
                    for patent_id in ids
                ],
                schema=_COMPACT_SCHEMA,
            )

    monkeypatch.setattr('patent_ate.extract.ray.data.from_arrow', capture_plan)
    monkeypatch.setattr('patent_ate.extract.PatentTermExtractor', _Recorder)
    if ray.is_initialized():
        ray.shutdown()
    ensure_local_ray()
    try:
        run_patent_extract(
            plan,
            extract_dir,
            tmp_path / 'ray_checkpoint',
            ate=AteSpec(),
            workers=2,
            batch_size=1,
        )
        assert submitted == [['p1', 'p2']]
        rows = [
            row for part in extract_dir.glob('*.parquet') for row in pq.read_table(part).to_pylist()
        ]
        by_id = {str(row[PATENT_ID_COLUMN]): row for row in rows}
        assert sorted(by_id) == ['p0', 'p1', 'p2']
        assert len(rows) == 3
        assert by_id['p0']['terms'] == seed_terms
        submitted.clear()
        run_patent_extract(
            plan,
            extract_dir,
            tmp_path / 'ray_checkpoint',
            ate=AteSpec(),
            workers=2,
            batch_size=1,
        )
        assert submitted == []
        rows = [
            row for part in extract_dir.glob('*.parquet') for row in pq.read_table(part).to_pylist()
        ]
        assert sorted(str(row[PATENT_ID_COLUMN]) for row in rows) == ['p0', 'p1', 'p2']
        assert len(rows) == 3
    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_extract_plan_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match='unique'):
        ExtractPlan(
            rows=(
                PatentPlanRow(patent_id='same', path='/a'),
                PatentPlanRow(patent_id='same', path='/b'),
            )
        )


def test_checkpoint_config_keeps_patent_id(tmp_path: Path) -> None:
    config = extract_checkpoint_config(tmp_path / 'ray_checkpoint')
    assert config.id_column == PATENT_ID_COLUMN
    assert config.delete_checkpoint_on_success is False
    context = configure_extract_context(tmp_path / 'ray_checkpoint')
    assert context.checkpoint_config.id_column == PATENT_ID_COLUMN
    assert context.retried_map_errors is True
    assert context.max_map_retries == 3
    assert context.max_errored_blocks == 0
    assert context.execution_no_progress_timeout_s == 1800


def test_cvalue_expression_compiles_containment_and_executes() -> None:
    rows = [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': [
                {'term': 'coil spring', 'frequency': 2, 'surfaces': ['coil spring']},
                {
                    'term': 'coil spring assembly',
                    'frequency': 1,
                    'surfaces': ['coil spring assembly'],
                },
            ],
        }
    ]
    patents = ibis.memtable(pa.Table.from_pylist(rows, schema=_COMPACT_SCHEMA))
    stats = term_stats(patents)
    stats_sql = ibis.to_sql(stats)
    surfaces_sql = ibis.to_sql(surfaces(patents))
    compact_stats = ibis.memtable({
        'term': ['coil spring', 'coil spring assembly'],
        'tf': [2, 1],
        'df': [1, 1],
        'word_count': [2, 3],
    })
    contrib_sql = ibis.to_sql(parent_contributions(compact_stats, buckets=4, bucket=1))
    one_sql = ibis.to_sql(parent_contributions(compact_stats, buckets=1, bucket=0))
    scored_sql = ibis.to_sql(
        scored_terms(
            ibis.memtable({'term': ['coil spring'], 'tf': [2], 'df': [1], 'word_count': [2]}),
            ibis.memtable({'child': ['coil spring'], 'p_ta': [1], 'sum_parent_tf': [1.0]}),
        )
    )
    keys_sql = ibis.to_sql(
        cvalue_keys(
            ibis.memtable({'term': ['coil spring'], 'df': [1], 'c_value': [1.0]}),
            ibis.memtable({'term': ['coil spring'], 'key': ['coil spring']}),
        )
    )
    assert 'UNNEST' in stats_sql
    assert 'RANGE' not in stats_sql.upper()
    assert 'REGEXP_EXTRACT_ALL' not in stats_sql.upper()
    assert 'UNNEST' in surfaces_sql
    assert 'RANGE' not in surfaces_sql.upper()
    assert 'REGEXP_EXTRACT_ALL' not in surfaces_sql.upper()
    assert 'RANGE' in contrib_sql.upper()
    assert 'REGEXP_EXTRACT_ALL' in contrib_sql.upper()
    assert JATE_CHUNK_PATTERN in contrib_sql or JATE_WORD_CLASS in contrib_sql
    assert r'\w+|\W+' not in contrib_sql
    assert 'REGEXP_ESCAPE' in contrib_sql.upper()
    assert contrib_sql.upper().index('WINDOW') < contrib_sql.upper().index('REGEXP_ESCAPE')
    _assert_window_hash_sql(contrib_sql)
    assert 'HASH(' in contrib_sql.upper()
    assert '%' in contrib_sql
    assert 'GROUP BY' in contrib_sql.upper()
    assert re.search(r'(=|= )\s*1\b', contrib_sql)
    assert 'HASH(' in one_sql.upper()
    assert 'GROUP BY' in one_sql.upper()
    assert 'surfaces' not in contrib_sql.lower()
    assert 'RANGE' not in scored_sql.upper()
    assert 'REGEXP_EXTRACT_ALL' not in scored_sql.upper()
    assert 'UNNEST' not in scored_sql
    assert 'LN(' in scored_sql or 'LOG' in scored_sql
    assert 'RANGE' not in keys_sql.upper()
    assert 'REGEXP_EXTRACT_ALL' not in keys_sql.upper()
    expr = cvalue_expression(patents)
    frame = expr.to_polars()
    table = termhood_table(patents)
    assert 'coil spring' in set(frame['key'].to_list())
    assert 'coil spring assembly' in set(frame['key'].to_list())
    assert set(table.c_values) == set(frame['key'].to_list())
    assert table.total_docs == 1


def _patent_rows(terms_by_patent: list[list[tuple[str, int]]]) -> list[dict[str, Any]]:
    return [
        {
            PATENT_ID_COLUMN: f'p{index}',
            'n_docs': 1,
            'terms': [
                {'term': term, 'frequency': frequency, 'surfaces': [term]}
                for term, frequency in terms
            ],
        }
        for index, terms in enumerate(terms_by_patent)
    ]


def _explain_physical(expr: ibis.Table) -> str:
    backend = expr._find_backend(use_default=True)
    cursor = backend.raw_sql(f'EXPLAIN {expr.compile()}')
    return '\n'.join(str(row[-1]) for row in cursor.fetchall())


def _assert_window_hash_sql(sql: str) -> None:
    predicates = _JOIN_EQ_PREDICATE.findall(sql)
    assert predicates
    assert all('REGEXP' not in predicate.upper() for predicate in predicates)
    assert _WINDOW_EQ_TERM.search(sql)


def test_cvalue_parent_windows_match_jate_and_space_tokens() -> None:
    overlapping = _patent_rows([
        [('a a a', 4), ('a a', 2)],
        [('spring spring assembly', 1), ('spring assembly', 3), ('spring spring', 2)],
    ])
    longer = _patent_rows([
        [('coil spring assembly mount', 1), ('coil spring', 2), ('assembly mount', 1)],
    ])
    absent = _patent_rows([[('vehicle interior', 5)]])
    spaced = _patent_rows([[('coil  spring assembly', 1), ('coil spring', 2)]])
    hyphen = _patent_rows([
        [
            ('anti-lock brake system', 1),
            ('anti-lock brake', 2),
            ('brake system', 3),
            ('lock brake', 4),
            ('anti lock', 1),
            ('ach high-voltage source', 1),
            ('high-voltage source', 1),
            ('voltage source', 2),
        ]
    ])
    prefixed = _patent_rows([
        [
            ('pre-charge-coupled device camera extra', 1),
            ('charge-coupled device camera', 2),
            ('foo-bar-baz qux extra', 1),
            ('bar-baz qux', 2),
        ]
    ])
    slashed = _patent_rows([
        [
            ('pre-foo-bar extra pre-foo/bar extra', 1),
            ('foo/bar extra', 2),
            ('foo-bar extra', 1),
        ]
    ])
    dotted = _patent_rows([
        [
            ('pre-foo-bar extra pre-foo.bar extra', 1),
            ('foo.bar extra', 2),
        ]
    ])
    plused = _patent_rows([
        [('foo+bar baz extra', 1), ('foo+bar baz', 2), ('foo+bar', 1)],
    ])
    leading = _patent_rows([
        [('-anti-lock brake extra', 1), ('anti-lock brake', 2)],
    ])
    trailing = _patent_rows([
        [('brake system extra-', 1), ('brake system', 2)],
    ])
    meta = _patent_rows([
        [('foo(bar) baz extra', 1), ('bar) baz', 2)],
    ])
    underscore = _patent_rows([
        [('foo_bar extra word', 1), ('foo_bar extra', 2), ('foo', 3)],
    ])
    apostrophe = _patent_rows([
        [("don't stop extra", 1), ("don't stop", 2), ('stop extra', 1)],
    ])
    digits = _patent_rows([
        [('iso9001 extra word', 1), ('iso9001 extra', 2), ('iso', 3), ('9001 extra', 1)],
    ])
    for rows in (
        overlapping,
        longer,
        absent,
        spaced,
        hyphen,
        prefixed,
        slashed,
        dotted,
        plused,
        leading,
        trailing,
        meta,
        underscore,
        apostrophe,
        digits,
    ):
        table = termhood_table(ibis.memtable(pa.Table.from_pylist(rows, schema=_COMPACT_SCHEMA)))
        assert table.c_values == pytest.approx(_jate_table(rows))
        assert all(math.isfinite(value) for value in table.c_values.values())
    lone = termhood_table(ibis.memtable(pa.Table.from_pylist(absent, schema=_COMPACT_SCHEMA)))
    assert lone.c_values['vehicle interior'] == pytest.approx(math.log2(2.1) * 5)
    spaced_table = termhood_table(
        ibis.memtable(pa.Table.from_pylist(spaced, schema=_COMPACT_SCHEMA))
    )
    assert spaced_table.c_values['coil spring'] == pytest.approx(math.log2(2.1) * 2)

    def scored_parents(rows: list[dict[str, Any]]) -> tuple[ibis.Table, dict[str, set[str]]]:
        stats = (
            ibis
            .memtable(pa.Table.from_pylist(rows, schema=_COMPACT_SCHEMA))
            .unnest('terms')
            .unpack('terms')
            .group_by('term')
            .agg(tf=ibis._.frequency.sum())
            .mutate(word_count=jate_word_count(ibis._.term))
        )
        windows = parent_windows(stats)
        parents_of: dict[str, set[str]] = defaultdict(set)
        for record in (
            windows
            .join(stats, windows.window == stats.term)
            .filter(windows.parent_word_count > stats.word_count)
            .select(child=stats.term, parent=windows.parent)
            .to_polars()
            .iter_rows(named=True)
        ):
            parents_of[str(record['child'])].add(str(record['parent']))
        return windows, parents_of

    _, parents_of = scored_parents(hyphen)
    assert 'anti-lock brake system' in parents_of['lock brake']
    assert 'anti-lock brake system' in parents_of['brake system']
    assert 'anti-lock brake system' in parents_of['anti-lock brake']
    assert 'anti-lock brake system' not in parents_of['anti lock']
    assert 'high-voltage source' not in parents_of['voltage source']
    assert 'ach high-voltage source' in parents_of['voltage source']
    hyphen_table = termhood_table(
        ibis.memtable(pa.Table.from_pylist(hyphen, schema=_COMPACT_SCHEMA))
    )
    assert 'lock' not in hyphen_table.c_values
    prefix_windows, prefix_parents = scored_parents(prefixed)
    extracted = set(prefix_windows.to_polars()['window'].to_list())
    assert 'charge-coupled device camera' in extracted
    assert 'bar-baz qux' in extracted
    assert 'charge coupled device camera' not in extracted
    assert 'bar baz qux' not in extracted
    assert prefix_parents['charge-coupled device camera'] == {
        'pre-charge-coupled device camera extra',
    }
    assert prefix_parents['bar-baz qux'] == {'foo-bar-baz qux extra'}
    slash_parent = 'pre-foo-bar extra pre-foo/bar extra'
    slash_windows, slash_parents = scored_parents(slashed)
    extracted_slash = set(slash_windows.to_polars()['window'].to_list())
    assert {'foo-bar extra', 'foo/bar extra'} <= extracted_slash
    assert slash_parents['foo/bar extra'] == {slash_parent}
    assert slash_parents['foo-bar extra'] == {slash_parent}
    assert int(slash_windows.count().to_pyarrow().as_py()) <= 80
    _, dot_parents = scored_parents(dotted)
    assert dot_parents['foo.bar extra'] == {'pre-foo-bar extra pre-foo.bar extra'}
    plus_windows, plus_parents = scored_parents(plused)
    assert 'foo+bar' in set(plus_windows.to_polars()['window'].to_list())
    assert plus_parents['foo+bar'] == {'foo+bar baz extra', 'foo+bar baz'}
    _, lead_parents = scored_parents(leading)
    assert lead_parents['anti-lock brake'] == {'-anti-lock brake extra'}
    _, trail_parents = scored_parents(trailing)
    assert trail_parents['brake system'] == {'brake system extra-'}
    _, meta_parents = scored_parents(meta)
    assert meta_parents['bar) baz'] == {'foo(bar) baz extra'}
    _, under_parents = scored_parents(underscore)
    assert under_parents['foo_bar extra'] == {'foo_bar extra word'}
    assert 'foo' not in under_parents
    _, apo_parents = scored_parents(apostrophe)
    assert apo_parents["don't stop"] == {"don't stop extra"}
    assert apo_parents['stop extra'] == {"don't stop extra"}
    _, digit_parents = scored_parents(digits)
    assert digit_parents['iso9001 extra'] == {'iso9001 extra word'}
    assert 'iso' not in digit_parents
    assert '9001 extra' not in digit_parents


def test_cvalue_parent_join_explains_as_hash_equality() -> None:
    rows = _patent_rows([
        [(f't{index:03d} extra word', 1), (f't{index:03d}', 2)] for index in range(80)
    ])
    backend = ibis.duckdb.connect()
    try:
        terms = [str(item['term']) for patent in rows for item in patent['terms']]
        tfs = [int(item['frequency']) for patent in rows for item in patent['terms']]
        stats = backend.create_table(
            'stats',
            pa.table({
                'term': terms,
                'tf': tfs,
                'df': [1] * len(terms),
                'word_count': [len(term.split()) for term in terms],
            }),
        )
        contrib = parent_contributions(stats)
        plan = _explain_physical(contrib).upper()
        sql = ibis.to_sql(contrib)
    finally:
        backend.disconnect()
    collapsed = re.sub(r'\s+', ' ', plan).replace('"', '')
    assert 'HASH_JOIN' in collapsed
    assert re.search(r'(WINDOW\s*=\s*TERM|TERM\s*=\s*WINDOW)', collapsed)
    assert 'CONDITIONS:' in collapsed
    assert 'NESTED_LOOP_JOIN' not in collapsed
    assert 'CROSS_PRODUCT' not in collapsed
    _assert_window_hash_sql(sql)
    assert 'HASH(' in sql.upper()
    assert '%' in sql
    assert 'GROUP BY' in sql.upper()
    assert 'REGEXP_EXTRACT_ALL' in sql.upper()
    assert 'REGEXP_ESCAPE' in sql.upper()
    assert JATE_CHUNK_PATTERN in sql or JATE_WORD_CLASS in sql
    assert r'\w+|\W+' not in sql


def test_parent_windows_cardinality_follows_candidate_span_lengths() -> None:
    width = 6
    parents = 40
    long_terms = [
        ' '.join(f't{index:02d}w{offset}' for offset in range(width)) for index in range(parents)
    ]
    child_terms = [f'child{index:02d} extra' for index in range(8)]
    terms = [*long_terms, *child_terms]
    backend = ibis.duckdb.connect()
    try:
        stats = backend.create_table(
            'stats',
            pa.table({'term': terms, 'tf': [1] * len(terms)}),
        ).mutate(word_count=jate_word_count(ibis._.term))
        lengths = candidate_span_lengths(stats)
        windows = parent_windows(stats, lengths)
        counted = int(windows.count().to_pyarrow().as_py())
        hyphen_terms = [
            *[f'pre-fix{index} extra word' for index in range(8)],
            *[f'fix{index} extra' for index in range(8)],
        ]
        hyphen_stats = backend.create_table(
            'hyphen_stats',
            pa.table({'term': hyphen_terms, 'tf': [1] * len(hyphen_terms)}),
        ).mutate(word_count=jate_word_count(ibis._.term))
        hyphen_windows = parent_windows(hyphen_stats, candidate_span_lengths(hyphen_stats))
        hyphen_counted = int(hyphen_windows.count().to_pyarrow().as_py())
        joined = (
            windows
            .join(stats, windows.window == stats.term)
            .filter(windows.parent_word_count > stats.word_count)
            .group_by(stats.term)
            .agg(n=ibis._.count())
        )
        hyphen_joined = hyphen_windows.join(
            hyphen_stats,
            hyphen_windows.window == hyphen_stats.term,
        )
        plan = _explain_physical(joined).upper()
        hyphen_plan = _explain_physical(hyphen_joined).upper()
        joined_sql = ibis.to_sql(joined)
        hyphen_sql = ibis.to_sql(hyphen_joined)
        window_sql = ibis.to_sql(windows)
    finally:
        backend.disconnect()
    assert counted == parents * (2 * width - 1 - 3 + 1)
    assert hyphen_counted > 0
    assert hyphen_counted < 8 * (4 * 5 // 2 - 1)
    assert 'SPAN_N' in window_sql.upper()
    assert window_sql.upper().index('SPAN_N') < window_sql.upper().index('RANGE')
    collapsed = re.sub(r'\s+', ' ', plan)
    assert 'HASH_JOIN' in collapsed
    assert 'WINDOW = TERM' in collapsed
    assert 'NESTED_LOOP_JOIN' not in collapsed
    assert 'CROSS_PRODUCT' not in collapsed
    _assert_window_hash_sql(joined_sql)
    assert 'WINDOW = TERM' in re.sub(r'\s+', ' ', hyphen_plan)
    assert 'NESTED_LOOP_JOIN' not in hyphen_plan
    assert 'CROSS_PRODUCT' not in hyphen_plan
    _assert_window_hash_sql(hyphen_sql)


def test_score_term_parquet_releases_connection_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[object] = []

    class _Backend:
        def read_parquet(self, _files: object) -> object:
            raise RuntimeError('score failed')

        def disconnect(self) -> None:
            released.append(True)

    monkeypatch.setattr(
        'patent_ate.cvalue.exec.ibis.duckdb.connect',
        lambda **_kwargs: _Backend(),
    )
    _write_compact(
        tmp_path / 'part-0.parquet',
        [
            {
                PATENT_ID_COLUMN: 'p1',
                'n_docs': 1,
                'terms': [
                    {'term': 'coil spring', 'frequency': 1, 'surfaces': ['coil spring']},
                ],
            }
        ],
    )
    with pytest.raises(RuntimeError, match='score failed'):
        score_term_parquet(tmp_path, ate=AteSpec(parent_buckets=1), temp_dir=tmp_path / 'duckdb')
    assert released == [True]


def test_normalized_key_matches_python_normalize_surface() -> None:
    samples = (
        '  (Coil Spring).  ',
        'Ġfoo##',
        '▁bar',
        '',
        'anti-lock',
    )
    table = ibis.memtable({'s': list(samples)})
    got = table.select(key=normalized_key(table.s)).to_polars()['key'].to_list()
    assert got == [normalize_surface(sample) for sample in samples]


def test_score_term_parquet_applies_ibis_duckdb_connect_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    real_connect = ibis.duckdb.connect

    def wrap_connect(**kwargs: object) -> object:
        captured.update(kwargs)
        return real_connect(**kwargs)

    monkeypatch.setattr('patent_ate.cvalue.exec.ibis.duckdb.connect', wrap_connect)
    rows = [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': [{'term': 'coil spring', 'frequency': 1, 'surfaces': ['coil spring']}],
        }
    ]
    _write_compact(tmp_path / 'part-0.parquet', rows)
    ate = AteSpec(duckdb_memory='256MB', duckdb_threads=8, parent_buckets=1)
    spill = tmp_path / 'duckdb'
    store, c_values, _freqs = _store_facts(score_term_parquet(tmp_path, ate=ate, temp_dir=spill))
    assert captured['memory_limit'] == ate.duckdb_memory == '256MB'
    assert captured['threads'] == ate.duckdb_threads == 8
    assert captured['temp_directory'] == str(spill)
    assert captured['preserve_insertion_order'] is False
    assert store.meta.total_docs == 1
    assert 'coil spring' in c_values


def test_corpus_termhood_scores_complete_extract_without_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / f'p{index}.json' for index in range(2))
    for path in paths:
        path.write_text('{}', encoding='utf-8')
    plan = ExtractPlan.from_paths(paths)
    plan_path = plan.write_parquet(tmp_path / PLAN_FILENAME)
    plan_bytes = plan_path.read_bytes()
    extract_dir = tmp_path / EXTRACT_DIRNAME
    _write_compact(
        extract_dir / 'part-0.parquet',
        [
            {
                PATENT_ID_COLUMN: row.patent_id,
                'n_docs': 1,
                'terms': [{'term': 'coil spring', 'frequency': 1, 'surfaces': ['coil spring']}],
            }
            for row in plan.rows
        ],
    )
    extract_bytes = (extract_dir / 'part-0.parquet').read_bytes()
    scored: list[Path] = []
    extract_calls: list[object] = []
    arrow_calls: list[object] = []

    def capture_score(*args: object, **kwargs: object) -> Path:
        scored.append(score_term_parquet(*args, **kwargs))
        return scored[-1]

    def forbid_extract(*_args: object, **_kwargs: object) -> None:
        extract_calls.append(True)

    def forbid_arrow(*_args: object, **_kwargs: object) -> object:
        arrow_calls.append(True)
        raise AssertionError('from_arrow must not run on a complete extract')

    monkeypatch.setattr('patent_ate.extract.score_term_parquet', capture_score)
    monkeypatch.setattr('patent_ate.extract.run_patent_extract', forbid_extract)
    monkeypatch.setattr('patent_ate.extract.ray.data.from_arrow', forbid_arrow)
    ate = AteSpec(duckdb_memory='256MB', duckdb_threads=8, parent_buckets=1)
    output = corpus_termhood(paths, ate=ate, chunk_size=1, workers=2, artifact_dir=tmp_path)
    assert scored
    assert output == scored[0]
    assert extract_calls == []
    assert arrow_calls == []
    assert plan_path.read_bytes() == plan_bytes
    assert (extract_dir / 'part-0.parquet').read_bytes() == extract_bytes
    assert extract_ids(extract_dir) == plan.ids


def test_score_term_parquet_matches_jate_cvalue(tmp_path: Path) -> None:
    rows = [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': [
                {'term': 'coil spring', 'frequency': 2, 'surfaces': ['coil spring']},
                {
                    'term': 'coil spring assembly',
                    'frequency': 1,
                    'surfaces': ['coil spring assembly'],
                },
            ],
        },
        {
            PATENT_ID_COLUMN: 'p2',
            'n_docs': 1,
            'terms': [
                {
                    'term': 'coil spring assembly',
                    'frequency': 1,
                    'surfaces': ['Coil Spring Assembly'],
                },
                {'term': 'vehicle interior', 'frequency': 1, 'surfaces': ['vehicle interior']},
            ],
        },
    ]
    _write_compact(tmp_path / 'part-0.parquet', rows)
    store, c_values, _freqs = _store_facts(
        score_term_parquet(tmp_path, ate=AteSpec(parent_buckets=1), temp_dir=tmp_path / 'duckdb')
    )
    expected = _jate_table(rows)
    assert store.meta.total_docs == 2
    assert set(c_values) == set(expected)
    for key, value in expected.items():
        assert c_values[key] == pytest.approx(value)


@given(
    corpus=st.lists(
        st.lists(st.sampled_from(_PHRASES), min_size=1, max_size=4, unique=True),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=20, deadline=None)
def test_generated_corpus_cvalue_matches_jate(corpus: list[list[str]]) -> None:
    rows = [
        {
            PATENT_ID_COLUMN: f'p{index}',
            'n_docs': 1,
            'terms': [
                {'term': phrase, 'frequency': 1 + index % 2, 'surfaces': [phrase]}
                for phrase in phrases
            ],
        }
        for index, phrases in enumerate(corpus)
    ]
    with TemporaryDirectory() as tmp:
        target = Path(tmp)
        _write_compact(target / 'part-0.parquet', rows)
        store, c_values, _freqs = _store_facts(
            score_term_parquet(target, ate=AteSpec(parent_buckets=1), temp_dir=target / 'duckdb')
        )
        expected = _jate_table(rows)
        assert store.meta.total_docs == len(rows)
        assert set(c_values) == set(expected)
        for key, value in expected.items():
            assert c_values[key] == pytest.approx(value)


def test_extractor_preserves_patent_ids_and_clips(monkeypatch: pytest.MonkeyPatch) -> None:
    ate = AteSpec()
    extractor = PatentTermExtractor(ate.spacy_model, pipe_docs=1, sentence_group=32)
    original = extractor.nlp.max_length
    extractor.nlp.max_length = 32
    monkeypatch.setattr(
        'patent_ate.extract.patent_rows',
        lambda _paths: (('x' * 80,), ('1',)),
    )
    try:
        batch = pa.table({PATENT_ID_COLUMN: ['patent-1']})
        out = extractor(batch)
    finally:
        extractor.nlp.max_length = original
    assert out.column(PATENT_ID_COLUMN).to_pylist() == ['patent-1']
    assert out.column('n_docs').to_pylist() == [1]


def test_extractor_on_fixture_keeps_stable_ids() -> None:
    paths, n_pool = sample_json_paths(_JSON_FIXTURES, limit=2, seed=0)
    assert n_pool >= 1
    plan = ExtractPlan.from_paths(paths)
    extractor = PatentTermExtractor(AteSpec().spacy_model, pipe_docs=1, sentence_group=32)
    batch = pa.table({
        PATENT_ID_COLUMN: [row.patent_id for row in plan.rows],
    })
    out = extractor(batch)
    assert out.column(PATENT_ID_COLUMN).to_pylist() == [row.patent_id for row in plan.rows]
    assert out.num_rows == len(plan.rows)


def test_default_sentence_group_frequencies_match_whole_document_jate() -> None:
    paths = (
        _JSON_FIXTURES / '13817165.json',
        _JSON_FIXTURES / '14111139.json',
    )
    ate = AteSpec()
    assert ate.sentence_group == 32
    extractor = PatentTermExtractor(ate.spacy_model, pipe_docs=1, sentence_group=ate.sentence_group)
    plan = ExtractPlan.from_paths(paths)
    texts, _apps = patent_rows(tuple(Path(row.patent_id) for row in plan.rows))
    compact = {
        row[PATENT_ID_COLUMN]: row
        for row in extractor(
            pa.table({PATENT_ID_COLUMN: [row.patent_id for row in plan.rows]})
        ).to_pylist()
    }
    named = ('working phase', 'associated series resistor')
    for row, text in zip(plan.rows, texts, strict=True):
        parsed = extractor.nlp(text)
        assert len(tuple(parsed.sents)) > ate.sentence_group
        whole = {
            cand.normalized_form.lower(): sum(len(pos) for pos in cand.doc_positions.values())
            for cand in JateDraw.extract((Document(doc_id=row.patent_id, content=text),), (parsed,))
        }
        got_row = compact[row.patent_id]
        assert got_row['n_docs'] == 1
        got = {item['term']: int(item['frequency']) for item in got_row['terms']}
        assert got == whole
        for key in named:
            if key in whole:
                assert got[key] == whole[key]


def test_sentence_groups_stay_bounded_on_long_patent() -> None:
    ate = AteSpec()
    extractor = PatentTermExtractor(ate.spacy_model, pipe_docs=1, sentence_group=ate.sentence_group)
    text = patent_rows((_JSON_FIXTURES / '14111139.json',))[0][0]
    doc = extractor.nlp(text)
    pieces = extractor._sentence_groups(doc)
    assert len(tuple(doc.sents)) > ate.sentence_group
    assert len(pieces) > 1
    assert all(len(tuple(piece.sents)) <= ate.sentence_group for piece in pieces)


def test_corpus_termhood_matches_jate_and_resumes(tmp_path: Path) -> None:
    paths, n_pool = sample_json_paths(_JSON_FIXTURES, limit=2, seed=0)
    assert n_pool >= 1
    ate = AteSpec(cpu_job_width=1, pipe_docs=1, sentence_group=100_000)
    first, c_values, freqs = _store_facts(
        corpus_termhood(paths, ate=ate, chunk_size=1, workers=1, artifact_dir=tmp_path)
    )
    extract_dir = tmp_path / EXTRACT_DIRNAME
    compact_rows = [
        row for part in extract_dir.glob('*.parquet') for row in pq.read_table(part).to_pylist()
    ]
    expected = _jate_table(compact_rows)
    assert first.meta.total_docs == sum(int(row['n_docs']) for row in compact_rows)
    assert set(c_values) == set(expected)
    assert c_values == pytest.approx(expected)
    assert extract_ids(extract_dir) == ExtractPlan.from_paths(paths).ids
    _second, again_c, again_df = _store_facts(
        corpus_termhood(paths, ate=ate, chunk_size=1, workers=1, artifact_dir=tmp_path)
    )
    assert again_c == c_values
    assert again_df == freqs

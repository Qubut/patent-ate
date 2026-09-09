"""Committed termhood Parquet plus generation manifest."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from patent_ate.cvalue import score_term_parquet
from patent_ate.extract import write_termhood
from patent_ate.plan import PATENT_ID_COLUMN
from patent_ate.spec import AteSpec
from patent_ate.termhood import (
    TERMHOOD_META_NAME,
    TERMHOOD_PARQUET_NAME,
    TERMHOOD_SCHEMA_VERSION,
    TermhoodIndex,
    TermhoodMeta,
    TermhoodStore,
    TermhoodTable,
)

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


def _write_extract(path: Path) -> Path:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    PATENT_ID_COLUMN: 'p1',
                    'n_docs': 1,
                    'terms': [
                        {'term': 'coil spring', 'frequency': 2, 'surfaces': ['coil spring']},
                    ],
                }
            ],
            schema=_COMPACT_SCHEMA,
        ),
        path,
    )
    return path


def _write_facts(path: Path, rows: list[dict[str, Any]]) -> Path:
    pl.DataFrame(
        rows,
        schema={'key': pl.String, 'c_value': pl.Float64, 'df': pl.Int64},
    ).write_parquet(path)
    return path


def _one_key_table() -> TermhoodTable:
    return TermhoodTable(c_values={'a': 1.0}, document_frequency={'a': 1}, total_docs=1)


def _sample_table() -> TermhoodTable:
    keys = tuple(f'phrase {index:02d}' for index in range(50))
    return TermhoodTable(
        c_values={key: 1.0 + float(index) for index, key in enumerate(keys)},
        document_frequency={key: 1 + (index % 5) for index, key in enumerate(keys)},
        total_docs=10_000,
    )


def test_parquet_round_trip_matches_in_memory_table(tmp_path: Path) -> None:
    table = TermhoodTable(
        c_values={'coil spring': 3.5, 'vehicle interior': 1.25},
        document_frequency={'coil spring': 2, 'vehicle interior': 4},
        total_docs=12,
    )
    store = TermhoodStore.open(write_termhood(table, tmp_path))
    index = TermhoodIndex.from_store(store)
    assert store.meta.schema_version == TERMHOOD_SCHEMA_VERSION
    assert store.meta.total_docs == 12
    assert store.meta.n_keys == 2
    assert store.root == tmp_path
    assert store.parquet.name == f'termhood.{store.meta.generation}.parquet'
    assert index.scores_for(('coil spring', 'missing', 'vehicle interior')) == table.scores_for((
        'coil spring',
        'missing',
        'vehicle interior',
    ))
    frame = pl.scan_parquet(store.parquet).collect()
    assert frame.columns == ['key', 'c_value', 'df']
    assert 'score' not in frame.columns


def test_legacy_json_termhood_still_loads(tmp_path: Path) -> None:
    payload = {
        'c_values': {'coil spring': 3.0},
        'document_frequency': {'coil spring': 2},
        'total_docs': 10,
        'scores': {'coil spring': 1.0},
    }
    path = tmp_path / 'termhood.json'
    _ = path.write_text(json.dumps(payload) + '\n', encoding='utf-8')
    loaded = TermhoodTable.load(json.loads(path.read_text(encoding='utf-8')))
    assert loaded.c_values['coil spring'] == pytest.approx(3.0)
    assert loaded.total_docs == 10
    expected = TermhoodTable.model_validate(payload).score('coil spring')
    assert loaded.score('coil spring') == pytest.approx(expected)


def test_open_refuses_missing_meta(tmp_path: Path) -> None:
    write_termhood(_one_key_table(), tmp_path)
    (tmp_path / TERMHOOD_META_NAME).unlink()
    with pytest.raises(ValueError, match='metadata is missing'):
        TermhoodStore.open(tmp_path)


def test_open_refuses_missing_parquet(tmp_path: Path) -> None:
    store = TermhoodStore.open(write_termhood(_one_key_table(), tmp_path))
    store.parquet.unlink()
    with pytest.raises(ValueError, match='parquet is missing'):
        TermhoodStore.open(tmp_path)


def test_open_refuses_n_keys_mismatch(tmp_path: Path) -> None:
    store = TermhoodStore.open(write_termhood(_one_key_table(), tmp_path))
    _ = (tmp_path / TERMHOOD_META_NAME).write_text(
        store.meta.model_copy(update={'n_keys': 9}).model_dump_json() + '\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='does not match'):
        TermhoodStore.open(tmp_path)


def test_open_refuses_unsupported_schema_version(tmp_path: Path) -> None:
    store = TermhoodStore.open(write_termhood(_one_key_table(), tmp_path))
    _ = (tmp_path / TERMHOOD_META_NAME).write_text(
        store.meta.model_copy(update={'schema_version': 99}).model_dump_json() + '\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='does not match'):
        TermhoodStore.open(tmp_path)


def test_inspect_refuses_duplicate_nonfinite_negative_and_empty_keys(tmp_path: Path) -> None:
    cases = (
        [{'key': 'a', 'c_value': 1.0, 'df': 1}, {'key': 'a', 'c_value': 2.0, 'df': 1}],
        [{'key': 'a', 'c_value': math.nan, 'df': 1}],
        [{'key': 'a', 'c_value': 1.0, 'df': -1}],
        [{'key': '', 'c_value': 1.0, 'df': 1}],
    )
    for index, rows in enumerate(cases):
        path = tmp_path / f'bad-{index}.parquet'
        _write_facts(path, rows)
        with pytest.raises(ValueError, match='termhood parquet'):
            TermhoodStore.inspect_parquet(path)


def test_retry_clears_owned_partials(tmp_path: Path) -> None:
    leftover = tmp_path / f'{TERMHOOD_PARQUET_NAME}.partial'
    meta_left = tmp_path / f'{TERMHOOD_META_NAME}.partial'
    _ = leftover.write_text('stale', encoding='utf-8')
    _ = meta_left.write_text('stale', encoding='utf-8')
    write_termhood(_one_key_table(), tmp_path)
    assert not leftover.exists()
    assert not meta_left.exists()
    assert TermhoodStore.open(tmp_path).meta.n_keys == 1


def test_score_term_parquet_sinks_without_termhood_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dumped: list[str] = []
    _write_extract(tmp_path / 'part-0.parquet')

    def forbid_table(_patents: object) -> TermhoodTable:
        dumped.append('termhood_table')
        raise AssertionError('corpus score must not materialize TermhoodTable')

    monkeypatch.setattr('patent_ate.cvalue.algebra.termhood_table', forbid_table)
    monkeypatch.setattr(
        TermhoodTable,
        'model_dump',
        lambda self, **_kwargs: dumped.append('model_dump') or {},
    )
    path = score_term_parquet(tmp_path, ate=AteSpec(parent_buckets=1), temp_dir=tmp_path / 'duckdb')
    assert path == tmp_path
    assert (tmp_path / TERMHOOD_META_NAME).is_file()
    assert dumped == []
    store = TermhoodStore.open(path)
    assert store.meta.total_docs == 1
    assert store.meta.n_keys >= 1


def test_index_scores_for_requested_keys_only(tmp_path: Path) -> None:
    table = _sample_table()
    store = TermhoodStore.open(write_termhood(table, tmp_path))
    index = TermhoodIndex.from_store(store)
    wanted = ('phrase 00', 'phrase 07', 'absent')
    got = index.scores_for(wanted)
    assert tuple(got) == wanted
    assert got == table.scores_for(wanted)
    assert index.score('phrase 07') == table.score('phrase 07')
    assert not hasattr(TermhoodStore, 'scores_for')


def test_report_ranks_caps_collect_height(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _sample_table()
    store = TermhoodStore.open(write_termhood(table, tmp_path))
    heights: list[int] = []
    original = pl.LazyFrame.collect

    def collect(self: pl.LazyFrame, *args: object, **kwargs: object) -> pl.DataFrame:
        frame = original(self, *args, **kwargs)
        heights.append(frame.height)
        return frame

    monkeypatch.setattr(pl.LazyFrame, 'collect', collect)
    by_score, by_docs, n_unique = store.report_ranks(40)
    assert heights
    assert max(heights) <= 40
    assert n_unique == len(table.scores)
    assert len(by_score) <= 40
    assert len(by_docs) <= 40
    assert by_score[0][1] >= by_score[-1][1]
    assert by_docs[0][2] >= by_docs[-1][2]


def test_index_scores_for_does_not_rescan_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _sample_table()
    store = TermhoodStore.open(write_termhood(table, tmp_path))
    scans: list[object] = []
    original = pl.scan_parquet

    def spy(source: object, *args: object, **kwargs: object) -> pl.LazyFrame:
        scans.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pl, 'scan_parquet', spy)
    index = TermhoodIndex.from_store(store)
    built = len(scans)
    assert built >= 1
    lookups = tuple(index.scores_for(('phrase 00', 'phrase 07', 'absent')) for _ in range(8))
    assert lookups[-1]['absent'] == pytest.approx(0.0)
    assert lookups[-1]['phrase 00'] == table.score('phrase 00')
    assert len(scans) == built


def test_index_zeros_stops_and_missing_keys(tmp_path: Path) -> None:
    table = TermhoodTable(
        c_values={'coil spring': 3.0, 'comprising': 9.0},
        document_frequency={'coil spring': 2, 'comprising': 2},
        total_docs=10,
    )
    index = TermhoodIndex.from_store(TermhoodStore.open(write_termhood(table, tmp_path)))
    got = index.scores_for(('coil spring', 'comprising', 'missing'))
    assert got == table.scores_for(('coil spring', 'comprising', 'missing'))
    assert got['comprising'] == pytest.approx(0.0)
    assert got['missing'] == pytest.approx(0.0)
    assert got['coil spring'] > 0.0


def test_crash_after_data_before_manifest_keeps_old_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = TermhoodTable(c_values={'a': 1.0}, document_frequency={'a': 1}, total_docs=3)
    old = TermhoodStore.open(write_termhood(first, tmp_path))
    original = Path.replace

    def boom(self: Path, target: Path, *args: object, **kwargs: object) -> Path:
        if Path(target).name == TERMHOOD_META_NAME:
            raise OSError('simulated crash')
        return original(self, target, *args, **kwargs)

    monkeypatch.setattr(Path, 'replace', boom)
    with pytest.raises(OSError, match='simulated crash'):
        write_termhood(
            TermhoodTable(c_values={'b': 2.0}, document_frequency={'b': 1}, total_docs=9),
            tmp_path,
        )
    still = TermhoodStore.open(tmp_path)
    assert still.meta.generation == old.meta.generation
    assert still.meta.total_docs == 3
    assert still.parquet.is_file()
    assert TermhoodIndex.from_store(still).score('a') == first.score('a')


def test_same_n_keys_changed_total_docs_cannot_mix(tmp_path: Path) -> None:
    facts = TermhoodTable(
        c_values={'coil spring': 1.0},
        document_frequency={'coil spring': 1},
        total_docs=10,
    )
    old = TermhoodStore.open(write_termhood(facts, tmp_path))
    write_termhood(
        TermhoodTable(
            c_values={'coil spring': 1.0},
            document_frequency={'coil spring': 1},
            total_docs=99,
        ),
        tmp_path,
    )
    new = TermhoodStore.open(tmp_path)
    assert old.meta.total_docs == 10
    assert new.meta.total_docs == 99
    assert old.meta.n_keys == new.meta.n_keys
    assert old.parquet.is_file()
    assert new.parquet.is_file()
    assert old.parquet != new.parquet
    assert TermhoodIndex.from_store(old).score('coil spring') != TermhoodIndex.from_store(
        new
    ).score('coil spring')


def test_open_refuses_corrupt_manifest(tmp_path: Path) -> None:
    write_termhood(_one_key_table(), tmp_path)
    _ = (tmp_path / TERMHOOD_META_NAME).write_text('{not-json', encoding='utf-8')
    with pytest.raises(ValueError, match='metadata is invalid'):
        TermhoodStore.open(tmp_path)


def test_open_type_mismatch_does_not_mutate_committed(tmp_path: Path) -> None:
    generation = 'ab' * 16
    parquet = tmp_path / f'termhood.{generation}.parquet'
    pl.DataFrame(
        {'key': ['a'], 'c_value': [1.0], 'df': [1]},
        schema={'key': pl.String, 'c_value': pl.Float32, 'df': pl.Int64},
    ).write_parquet(parquet)
    _ = (tmp_path / TERMHOOD_META_NAME).write_text(
        TermhoodMeta(
            schema_version=TERMHOOD_SCHEMA_VERSION,
            total_docs=1,
            n_keys=1,
            generation=generation,
            data_file=parquet.name,
        ).model_dump_json()
        + '\n',
        encoding='utf-8',
    )
    before = parquet.stat()
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='schema is invalid'):
        TermhoodStore.open(tmp_path)
    after = parquet.stat()
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == digest


def test_clear_partials_touches_owned_partials_only(tmp_path: Path) -> None:
    store = TermhoodStore.open(write_termhood(_one_key_table(), tmp_path))
    leftover = tmp_path / f'{TERMHOOD_PARQUET_NAME}.partial'
    gen_partial = tmp_path / f'termhood.{"b" * 32}.parquet.partial'
    meta_left = tmp_path / f'{TERMHOOD_META_NAME}.partial'
    _ = leftover.write_text('stale', encoding='utf-8')
    _ = gen_partial.write_text('stale', encoding='utf-8')
    _ = meta_left.write_text('stale', encoding='utf-8')
    TermhoodStore.clear_partials(tmp_path)
    assert store.parquet.is_file()
    assert (tmp_path / TERMHOOD_META_NAME).is_file()
    assert not leftover.exists()
    assert not gen_partial.exists()
    assert not meta_left.exists()


def test_index_build_records_memory_projection(tmp_path: Path) -> None:
    count = 8_000
    store = TermhoodStore.write_frame(
        pl.DataFrame({
            'key': [f'phrase {index:05d}' for index in range(count)],
            'c_value': [1.0] * count,
            'df': [1] * count,
        }),
        tmp_path,
        total_docs=count,
    )
    started = time.perf_counter()
    index = TermhoodIndex.from_store(store)
    elapsed = time.perf_counter() - started
    nbytes = sys.getsizeof(index.scores) + sum(
        sys.getsizeof(key) + sys.getsizeof(value) for key, value in index.scores.items()
    )
    assert len(index.scores) == count
    assert index.scores_for(('missing',))['missing'] == pytest.approx(0.0)
    assert nbytes > 0
    assert elapsed >= 0.0
    bytes_per_key = nbytes / count
    _ = (20_000_000 * bytes_per_key, 40_000_000 * bytes_per_key)

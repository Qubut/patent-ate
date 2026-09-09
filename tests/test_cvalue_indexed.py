"""Indexed GSA containment, resume, and C-value equality."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ibis
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

import patent_ate.cvalue.exec as exec_mod
from patent_ate.cvalue import plan_term_score, score_term_parquet
from patent_ate.cvalue.algebra import indexed_interval_pairs
from patent_ate.cvalue.exec import ScoreStageExecutor
from patent_ate.cvalue.index import (
    KEY_INTERVALS_NAME,
    PREFIX,
    SA_COLOR_NAME,
    IndexedContainmentError,
    concat_key_text,
    exact_bounds,
    prefix_bounds,
    resume_or_write_pattern_parents,
    term_identity,
    write_interval_chunks,
    write_pattern_parents,
)
from patent_ate.cvalue.index import build_gsa as build_gsa_arrays
from patent_ate.cvalue.plan import (
    INDEXED_ARTIFACT_VERSION,
    INDEXED_CONTRIB_BATCH_PACKS,
    INDEXED_CONTRIB_BATCH_WORKERS,
    INDEXED_CONTRIB_CHUNKS,
    INDEXED_CONTRIB_DIRNAME,
    INDEXED_MAX_CHUNK_COST,
    INDEXED_MAX_INTERVALS_PER_CHUNK,
    INDEXED_WORKDIR,
    SCORED_CHUNK_DIRNAME,
)
from patent_ate.cvalue.store import (
    SCORE_STAGE_DIRNAME,
    IndexedMapsFingerprint,
    ScoreStageStore,
)
from patent_ate.cvalue.text import surface_key
from patent_ate.plan import PATENT_ID_COLUMN
from patent_ate.spec import AteSpec, IndexedContainmentSpec
from patent_ate.termhood import TermhoodStore

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


def _write_compact(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_COMPACT_SCHEMA), path)
    return path


def _terms(*phrases: str, frequency: int = 1) -> list[dict[str, Any]]:
    return [{'term': phrase, 'frequency': frequency, 'surfaces': [phrase]} for phrase in phrases]


def _one_patent(*phrases: str) -> list[dict[str, Any]]:
    return [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': _terms(*phrases),
        }
    ]


def _pairs_from_listing(work: Path) -> pl.DataFrame:
    intervals = pl.read_parquet(work / KEY_INTERVALS_NAME)
    sa_color = pl.read_parquet(work / SA_COLOR_NAME)
    identity = pl.read_parquet(work / 'term_identity.parquet')
    return (
        intervals
        .join_where(
            sa_color,
            pl.col('rank') >= pl.col('left'),
            pl.col('rank') < pl.col('right'),
        )
        .filter((pl.col('parent_color') >= 0) & (pl.col('parent_color') != pl.col('color')))
        .join(
            identity.select(parent_color=pl.col('color'), parent=pl.col('term')),
            on='parent_color',
        )
        .select('key', 'parent')
        .unique()
    )


def _indexed_ate(**overrides: Any) -> AteSpec:
    values = {
        'duckdb_memory': '256MB',
        'duckdb_threads': 2,
        'parent_buckets': 1,
        'containment': 'indexed',
        'indexed': IndexedContainmentSpec(),
    }
    values.update(overrides)
    return AteSpec(**values)


def _window_ate() -> AteSpec:
    return AteSpec(duckdb_memory='256MB', duckdb_threads=2, parent_buckets=1)


def _mixed_rows() -> list[dict[str, Any]]:
    return [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': _terms(
                'a b',
                'a b extra',
                'lock brake',
                'anti-lock brake extra',
                'coil spring',
                'coil spring assembly',
                'Coil Spring Assembly',
                'helix path',
                '\u03b1helix path extra',
                'nf\u03bab pathway',
                'nf\u03bab pathway extra',
            ),
        }
    ]


def test_default_containment_is_window() -> None:
    assert AteSpec().containment == 'window'
    assert AteSpec().indexed is None


def test_indexed_requires_settings() -> None:
    with pytest.raises(ValidationError, match='indexed settings'):
        AteSpec(containment='indexed')


def test_indexed_spec_defaults_to_gsa() -> None:
    spec = IndexedContainmentSpec()
    assert spec.engine == 'gsa'
    assert spec.prefix_bytes == 8
    assert INDEXED_ARTIFACT_VERSION == '3'


def test_indexed_refuses_other_prefix_width() -> None:
    with pytest.raises(ValidationError):
        IndexedContainmentSpec(prefix_bytes=7)


def test_short_long_boundary_is_seven_and_eight_bytes() -> None:
    short = term_identity(pl.DataFrame({'term': ['a bcdef']}))
    long = term_identity(pl.DataFrame({'term': ['ab cdefg']}))
    assert int(short['norm_nbytes'][0]) == 7
    assert str(short['tier'][0]) == 'short'
    assert int(long['norm_nbytes'][0]) == 8
    assert str(long['tier'][0]) == 'long'
    assert PREFIX.exclusive_upper == 8
    assert PREFIX.width == 8


def test_normalization_collision_fanout() -> None:
    frame = term_identity(pl.DataFrame({'term': ['foo bar', 'Foo Bar']}))
    assert frame.height == 2
    assert frame['key'].n_unique() == 1
    assert set(frame['tier'].to_list()) == {'short'}


def test_nul_in_key_fails_closed() -> None:
    identity = pl.DataFrame({
        'color': [0],
        'term': ['bad'],
        'key': ['ab\x00cd'],
        'norm_nbytes': [5],
        'tier': ['short'],
    })
    with pytest.raises(IndexedContainmentError, match='NUL'):
        concat_key_text(identity)


def test_gsa_lists_byte_parents_without_id_fanout(tmp_path: Path) -> None:
    identity = term_identity(pl.DataFrame({'term': ['aa', 'aaa', 'baa', 'zz']}))
    dest = tmp_path / 'pattern_parents.parquet'
    write_pattern_parents(identity, dest)
    assert not dest.is_file()
    pairs = _pairs_from_listing(tmp_path)
    assert set(pairs.columns) == {'key', 'parent'}
    assert pairs.height == pairs.select(['key', 'parent']).n_unique()
    keys = set(zip(pairs['key'].to_list(), pairs['parent'].to_list(), strict=True))
    assert ('aa', 'aaa') in keys
    assert ('aa', 'baa') in keys
    assert ('aa', 'aa') not in keys


def test_prefix_searchsorted_matches_known_interval(tmp_path: Path) -> None:
    identity = term_identity(pl.DataFrame({'term': ['aaa', 'aa', 'baa']}))
    arrays = build_gsa_arrays(concat_key_text(identity))
    bounds = exact_bounds(identity, arrays, prefix_bounds(identity, arrays))
    aa = bounds.frame.filter(pl.col('key') == 'aa')
    assert aa.height == 1
    width = int(aa['right'][0]) - int(aa['left'][0])
    assert width >= 3
    write_pattern_parents(identity, tmp_path / 'pattern_parents.parquet')
    listed = _pairs_from_listing(tmp_path)
    parents = set(listed.filter(pl.col('key') == 'aa')['parent'].to_list())
    assert parents == {'aaa', 'baa'}


def test_shared_eight_byte_prefix_keeps_true_parent_only(tmp_path: Path) -> None:
    identity = term_identity(
        pl.DataFrame({
            'term': [
                'abcdefgh tail one extra',
                'abcdefgh tail two extra',
                'abcdefgh tail one extra more',
                'unrelated phrase here',
            ]
        })
    )
    dest = tmp_path / 'pattern_parents.parquet'
    write_pattern_parents(identity, dest)
    pairs = set(zip(*_pairs_from_listing(tmp_path).select('key', 'parent'), strict=True))
    child = surface_key('abcdefgh tail one extra')
    assert (child, 'abcdefgh tail one extra more') in pairs
    assert (child, 'abcdefgh tail two extra') not in pairs


def test_pattern_parents_dedupe_without_parts(tmp_path: Path) -> None:
    identity = term_identity(pl.DataFrame({'term': ['a b', 'A B', 'a b extra', 'unrelated']}))
    dest = tmp_path / 'pattern_parents.parquet'
    write_pattern_parents(identity, dest)
    assert not dest.is_file()
    assert (tmp_path / KEY_INTERVALS_NAME).is_file()
    assert (tmp_path / SA_COLOR_NAME).is_file()
    assert not dest.with_name(f'{dest.stem}_parts').exists()
    pairs = _pairs_from_listing(tmp_path)
    assert set(pairs.columns) == {'key', 'parent'}
    assert pairs.height == pairs.select(['key', 'parent']).n_unique()
    assert pairs.height < identity.height * pairs.select('parent').n_unique()


def test_listing_persists_interval_maps_not_prefix_cartesian(tmp_path: Path) -> None:
    terms = [f'abcdefgh variant {index:04d} extra token' for index in range(40)]
    terms.append('abcdefgh variant 0000 extra token more')
    identity = term_identity(pl.DataFrame({'term': terms}))
    dest = tmp_path / 'pattern_parents.parquet'
    write_pattern_parents(identity, dest)
    assert not dest.is_file()
    intervals = pl.read_parquet(tmp_path / KEY_INTERVALS_NAME)
    sa_color = pl.read_parquet(tmp_path / SA_COLOR_NAME)
    assert set(intervals.columns) == {'color', 'key', 'left', 'right'}
    assert set(sa_color.columns) == {'rank', 'parent_color'}
    assert 'parent' not in intervals.columns
    assert intervals.height == identity.height
    assert sa_color.height == concat_key_text(identity).n
    assert intervals.select(['left', 'right']).n_unique() > 1
    assert intervals.height + sa_color.height < 10_000


def test_indexed_short_durable_pairs_are_verified_not_id_fanout(tmp_path: Path) -> None:
    sharing = ('ab cd', 'Ġab cd', 'Ab Cd', 'AB CD', 'ab CD', 'Ab cd')
    extract = tmp_path / 'extract'
    _write_compact(
        extract / 'part-0.parquet',
        [
            {
                PATENT_ID_COLUMN: 'p1',
                'n_docs': 1,
                'terms': _terms(
                    *sharing,
                    'ab cd extra',
                    'ab cd extra ab cd extra',
                    'ab cd extra term',
                    'unrelated long enough phrase',
                ),
            }
        ],
    )
    window_dest = tmp_path / 'window'
    indexed_dest = tmp_path / 'indexed'
    window_root = score_term_parquet(
        extract, ate=_window_ate(), temp_dir=tmp_path / 'wspill', artifact_dir=window_dest
    )
    indexed_root = score_term_parquet(
        extract,
        ate=_indexed_ate(),
        temp_dir=tmp_path / 'ispill',
        artifact_dir=indexed_dest,
    )
    work = indexed_dest / SCORE_STAGE_DIRNAME / INDEXED_WORKDIR
    assert not (work / 'short_candidates.parquet').is_file()
    assert not (work / 'long_candidates.parquet').is_file()
    assert not (work / 'candidate_pairs.parquet').is_file()
    assert not (work / 'pattern_parents.parquet').is_file()
    intervals = pl.read_parquet(work / KEY_INTERVALS_NAME)
    sa_color = pl.read_parquet(work / SA_COLOR_NAME)
    assert set(intervals.columns) == {'color', 'key', 'left', 'right'}
    assert set(sa_color.columns) == {'rank', 'parent_color'}
    assert intervals.height == intervals.select('color').n_unique()
    patterns = _pairs_from_listing(work)
    assert patterns.height == patterns.select(['key', 'parent']).n_unique()
    assert patterns.height < len(sharing) * patterns.select('parent').n_unique()
    contrib = pl.read_parquet(indexed_dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet')
    children = set(contrib['child'].to_list())
    assert 'ab cd' in children
    assert 'Ġab cd' not in children
    assert surface_key('Ġab cd') == surface_key('ab cd')
    window_keys = TermhoodStore.open(window_root)
    indexed_keys = TermhoodStore.open(indexed_root)
    window_frame = pl.read_parquet(window_keys.parquet).sort('key')
    indexed_frame = pl.read_parquet(indexed_keys.parquet).sort('key')
    assert window_frame.select('key').to_series().to_list() == (
        indexed_frame.select('key').to_series().to_list()
    )
    matched = zip(
        window_frame.iter_rows(named=True),
        indexed_frame.iter_rows(named=True),
        strict=True,
    )
    assert all(
        left['c_value'] == pytest.approx(right['c_value']) and int(left['df']) == int(right['df'])
        for left, right in matched
    )
    assert (
        pl
        .read_parquet(window_dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet')
        .sort('child')
        .to_dicts()
        == contrib.sort('child').to_dicts()
    )


def test_indexed_matches_window_cvalue_and_contrib(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _mixed_rows())
    window_dest = tmp_path / 'window'
    indexed_dest = tmp_path / 'indexed'
    window_root = score_term_parquet(
        extract, ate=_window_ate(), temp_dir=tmp_path / 'wspill', artifact_dir=window_dest
    )
    indexed_root = score_term_parquet(
        extract,
        ate=_indexed_ate(),
        temp_dir=tmp_path / 'ispill',
        artifact_dir=indexed_dest,
    )
    window_keys = TermhoodStore.open(window_root)
    indexed_keys = TermhoodStore.open(indexed_root)
    assert window_keys.meta.n_keys == indexed_keys.meta.n_keys
    window_frame = pl.read_parquet(window_keys.parquet).sort('key')
    indexed_frame = pl.read_parquet(indexed_keys.parquet).sort('key')
    assert window_frame.select('key').to_series().to_list() == (
        indexed_frame.select('key').to_series().to_list()
    )
    matched = zip(
        window_frame.iter_rows(named=True),
        indexed_frame.iter_rows(named=True),
        strict=True,
    )
    assert all(
        left['key'] == right['key']
        and left['c_value'] == pytest.approx(right['c_value'])
        and int(left['df']) == int(right['df'])
        for left, right in matched
    )
    window_contrib = pl.read_parquet(
        window_dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet'
    ).sort('child')
    indexed_contrib = pl.read_parquet(
        indexed_dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet'
    ).sort('child')
    assert window_contrib.sort('child').to_dicts() == indexed_contrib.sort('child').to_dicts()


def test_indexed_hyphen_and_unicode_verify(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _mixed_rows())
    dest = tmp_path / 'out'
    score_term_parquet(
        extract,
        ate=_indexed_ate(),
        temp_dir=tmp_path / 'spill',
        artifact_dir=dest,
    )
    contrib = pl.read_parquet(dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet')
    children = set(contrib['child'].to_list())
    assert 'lock brake' in children
    assert 'nf\u03bab pathway' in children
    assert 'helix path' not in children
    assert 'a b' in children


def test_indexed_self_excluded_by_word_count(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _one_patent('coil spring assembly'))
    dest = tmp_path / 'out'
    score_term_parquet(
        extract,
        ate=_indexed_ate(),
        temp_dir=tmp_path / 'spill',
        artifact_dir=dest,
    )
    contrib = dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet'
    frame = pl.read_parquet(contrib)
    assert frame.height == 0


def test_window_and_indexed_cannot_cross_resume(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _mixed_rows())
    dest = tmp_path / 'dest'
    score_term_parquet(extract, ate=_window_ate(), temp_dir=tmp_path / 'wspill', artifact_dir=dest)
    window_contrib = (dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet').stat()
    score_term_parquet(
        extract,
        ate=_indexed_ate(),
        temp_dir=tmp_path / 'ispill',
        artifact_dir=dest,
    )
    store = ScoreStageStore(dest)
    manifest = store.manifest_path.read_text(encoding='utf-8')
    assert '"containment":"indexed"' in manifest.replace(' ', '')
    assert '"gsa_engine":"gsa"' in manifest.replace(' ', '')
    assert (dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet').stat() != window_contrib


def test_indexed_plan_resume_keeps_matching_fingerprint(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _mixed_rows())
    dest = tmp_path / 'dest'
    ate = _indexed_ate()
    first = plan_term_score(extract, ate=ate, temp_dir=tmp_path / 'spill', artifact_dir=dest)
    second = plan_term_score(extract, ate=ate, temp_dir=tmp_path / 'spill', artifact_dir=dest)
    assert first.indexed is not None
    assert second.indexed is not None
    assert first.indexed == second.indexed
    assert first.indexed.artifact_version == '3'
    assert first.indexed.engine == 'gsa'
    assert second.resume_units is True


def test_themisto_plan_cannot_resume(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _mixed_rows())
    dest = tmp_path / 'dest'
    ate = _indexed_ate()
    first = plan_term_score(extract, ate=ate, temp_dir=tmp_path / 'spill', artifact_dir=dest)
    assert first.indexed is not None
    plan_path = dest / SCORE_STAGE_DIRNAME / 'indexed_containment_plan.json'
    plan_path.write_text(
        first.indexed.model_copy(update={'artifact_version': '2'}).model_dump_json() + '\n',
        encoding='utf-8',
    )
    again = plan_term_score(extract, ate=ate, temp_dir=tmp_path / 'spill', artifact_dir=dest)
    assert again.indexed is not None
    assert again.indexed.artifact_version == '3'
    assert again.resume_units is False


def test_surface_key_matches_encoder_identity() -> None:
    assert surface_key('ĠFoo Bar!') == 'foo bar'


def test_indexed_interval_pairs_filters_sa_color_rank_band(tmp_path: Path) -> None:
    sa_path = tmp_path / SA_COLOR_NAME
    ranks = list(range(30_000))
    pq.write_table(
        pa.table({
            'rank': pa.array(ranks, type=pa.int64()),
            'parent_color': pa.array(
                [1 if rank < 10_000 else 2 if rank < 20_000 else 3 for rank in ranks],
                type=pa.int32(),
            ),
        }),
        sa_path,
        row_group_size=10_000,
    )
    backend = ibis.duckdb.connect()
    try:
        identity = backend.create_table(
            'identity',
            {'color': [0, 1, 2, 3], 'term': ['child', 'out_lo', 'in_band', 'out_hi']},
        )
        intervals = backend.create_table(
            'intervals',
            {'color': [0], 'left': [0], 'right': [30_000]},
        )
        expr = indexed_interval_pairs(
            identity,
            intervals,
            backend.read_parquet(sa_path),
            rank_lo=10_000,
            rank_hi=20_000,
        )
        sql = ibis.to_sql(expr)
        plan = '\n'.join(
            str(row[-1]) for row in backend.raw_sql(f'EXPLAIN {expr.compile()}').fetchall()
        )
        pairs = expr.to_polars()
    finally:
        backend.disconnect()
    assert re.search(r'"rank"\s*>=\s*10000', sql)
    assert re.search(r'"rank"\s*<\s*20000', sql)
    assert '10000' in plan
    assert '20000' in plan
    assert set(pairs['parent'].to_list()) == {'in_band'}
    assert pairs.height == 1


def test_pattern_parents_resume_skips_listing(tmp_path: Path) -> None:
    identity = term_identity(pl.DataFrame({'term': ['a b', 'a b extra']}))
    dest = tmp_path / 'pattern_parents.parquet'
    write_pattern_parents(identity, dest)
    maps = (
        tmp_path / KEY_INTERVALS_NAME,
        tmp_path / SA_COLOR_NAME,
        tmp_path / 'term_identity.parquet',
    )
    mtimes = tuple(path.stat().st_mtime_ns for path in maps)
    assert resume_or_write_pattern_parents(identity, dest) is True
    assert tuple(path.stat().st_mtime_ns for path in maps) == mtimes
    assert not dest.is_file()


def test_indexed_score_resumes_maps_and_contrib_chunks(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _mixed_rows())
    dest = tmp_path / 'indexed'
    ate = _indexed_ate()
    score_term_parquet(extract, ate=ate, temp_dir=tmp_path / 'spill-a', artifact_dir=dest)
    work = dest / SCORE_STAGE_DIRNAME / INDEXED_WORKDIR
    maps = (
        work / KEY_INTERVALS_NAME,
        work / SA_COLOR_NAME,
        work / 'term_identity.parquet',
    )
    chunks = tuple(sorted((work / INDEXED_CONTRIB_DIRNAME).glob('*.parquet')))
    assert chunks
    map_mtime = tuple(path.stat().st_mtime_ns for path in maps)
    chunk_mtime = tuple(path.stat().st_mtime_ns for path in chunks)
    (dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet').unlink()
    score_term_parquet(extract, ate=ate, temp_dir=tmp_path / 'spill-b', artifact_dir=dest)
    assert tuple(path.stat().st_mtime_ns for path in maps) == map_mtime
    assert tuple(path.stat().st_mtime_ns for path in chunks) == chunk_mtime
    assert (dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet').is_file()


def test_indexed_contrib_batch_amortizes_connect_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unfinished packs share one DuckDB session per batch; finished files stay."""
    monkeypatch.setattr('patent_ate.cvalue.exec.INDEXED_MAX_INTERVALS_PER_CHUNK', 2)
    monkeypatch.setattr('patent_ate.cvalue.exec.INDEXED_CONTRIB_BATCH_PACKS', 2)
    batch_runs: list[object] = []
    real_run = ScoreStageExecutor.run

    def spy_run(self: ScoreStageExecutor, use: Any) -> Any:
        if 'write_indexed_chunk_batch' in getattr(use, '__qualname__', ''):
            batch_runs.append(use)
        return real_run(self, use)

    monkeypatch.setattr(ScoreStageExecutor, 'run', spy_run)
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _mixed_rows())
    dest = tmp_path / 'indexed'
    ate = _indexed_ate()
    with capture_logs() as logs:
        score_term_parquet(extract, ate=ate, temp_dir=tmp_path / 'spill-a', artifact_dir=dest)
    work = dest / SCORE_STAGE_DIRNAME / INDEXED_WORKDIR
    chunk_dir = work / INDEXED_CONTRIB_DIRNAME
    chunks = tuple(sorted(chunk_dir.glob('*.parquet')))
    assert len(chunks) >= 4
    batch_starts = tuple(
        entry
        for entry in logs
        if entry.get('stage') == 'parent_contrib_batch' and entry.get('phase') == 'start'
    )
    written = tuple(
        entry
        for entry in logs
        if entry.get('stage') == 'parent_contrib_chunk' and entry.get('resumed') is False
    )
    assert len(written) >= 4
    assert len(batch_starts) >= 2
    assert len(batch_starts) < len(written)
    assert len(batch_runs) == len(batch_starts)
    assert len(batch_runs) < len(written)
    assert max(int(entry['n_packs']) for entry in batch_starts) <= 2

    kept = chunks[0]
    kept_mtime = kept.stat().st_mtime_ns
    for path in chunks[1:]:
        path.unlink()
    (dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet').unlink()
    batch_runs.clear()
    with capture_logs() as resume_logs:
        score_term_parquet(extract, ate=ate, temp_dir=tmp_path / 'spill-b', artifact_dir=dest)
    assert kept.stat().st_mtime_ns == kept_mtime
    resumed = tuple(
        entry
        for entry in resume_logs
        if entry.get('stage') == 'parent_contrib_chunk'
        and entry.get('resumed') is True
        and entry.get('chunk_i') == 0
    )
    assert resumed
    pending_writes = tuple(
        entry
        for entry in resume_logs
        if entry.get('stage') == 'parent_contrib_chunk' and entry.get('resumed') is False
    )
    resume_batches = tuple(
        entry
        for entry in resume_logs
        if entry.get('stage') == 'parent_contrib_batch' and entry.get('phase') == 'start'
    )
    assert pending_writes
    assert len(resume_batches) < len(pending_writes)
    assert len(batch_runs) == len(resume_batches)
    assert len(batch_runs) < len(pending_writes)
    assert INDEXED_CONTRIB_BATCH_PACKS == 64
    assert INDEXED_CONTRIB_BATCH_WORKERS == 8
    assert (dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet').is_file()


def test_indexed_contrib_parallel_workers_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disjoint batch lanes run on separate DuckDB spills; finished packs stay."""
    monkeypatch.setattr('patent_ate.cvalue.exec.INDEXED_MAX_INTERVALS_PER_CHUNK', 2)
    monkeypatch.setattr('patent_ate.cvalue.exec.INDEXED_CONTRIB_BATCH_PACKS', 2)
    monkeypatch.setattr('patent_ate.cvalue.exec.INDEXED_CONTRIB_BATCH_WORKERS', 2)
    monkeypatch.setattr(exec_mod.os, 'cpu_count', lambda: 4)
    if hasattr(exec_mod.os, 'process_cpu_count'):
        monkeypatch.setattr(exec_mod.os, 'process_cpu_count', lambda: 4)
    lane_spills: list[str] = []
    real_init = ScoreStageExecutor.__init__

    def spy_init(self: ScoreStageExecutor, ate: AteSpec, temp_dir: Path) -> None:
        if temp_dir.name.startswith('contrib-worker-'):
            lane_spills.append(temp_dir.name)
        real_init(self, ate, temp_dir)

    monkeypatch.setattr(ScoreStageExecutor, '__init__', spy_init)
    submitted: list[object] = []
    real_submit = ThreadPoolExecutor.submit

    def spy_submit(self: ThreadPoolExecutor, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        submitted.append(fn)
        return real_submit(self, fn, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, 'submit', spy_submit)
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _mixed_rows())
    dest = tmp_path / 'indexed'
    ate = _indexed_ate()
    with capture_logs() as logs:
        score_term_parquet(extract, ate=ate, temp_dir=tmp_path / 'spill-a', artifact_dir=dest)
    chunk_dir = dest / SCORE_STAGE_DIRNAME / INDEXED_WORKDIR / INDEXED_CONTRIB_DIRNAME
    chunks = tuple(sorted(chunk_dir.glob('*.parquet')))
    assert len(chunks) >= 4
    starts = tuple(
        entry
        for entry in logs
        if entry.get('stage') == 'parent_contrib_batch' and entry.get('phase') == 'start'
    )
    assert len(starts) >= 2
    assert {int(entry['n_workers']) for entry in starts} == {2}
    assert {int(entry['worker_i']) for entry in starts} == {0, 1}
    assert set(lane_spills) == {'contrib-worker-0', 'contrib-worker-1'}
    assert len(submitted) == 2

    kept = chunks[0]
    kept_mtime = kept.stat().st_mtime_ns
    for path in chunks[1:]:
        path.unlink()
    (dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet').unlink()
    lane_spills.clear()
    submitted.clear()
    with capture_logs() as resume_logs:
        score_term_parquet(extract, ate=ate, temp_dir=tmp_path / 'spill-b', artifact_dir=dest)
    assert kept.stat().st_mtime_ns == kept_mtime
    resumed = tuple(
        entry
        for entry in resume_logs
        if entry.get('stage') == 'parent_contrib_chunk'
        and entry.get('resumed') is True
        and entry.get('chunk_i') == 0
    )
    assert resumed
    pending_writes = tuple(
        entry
        for entry in resume_logs
        if entry.get('stage') == 'parent_contrib_chunk' and entry.get('resumed') is False
    )
    assert pending_writes
    assert (dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet').is_file()


def _interval_rows(n: int, *, left0: int = 0) -> pl.DataFrame:
    left = list(range(left0, left0 + n))
    return pl.DataFrame({
        'color': list(range(n)),
        'key': [f'k{i}' for i in range(n)],
        'left': left,
        'right': [value + 1 for value in left],
    })


def test_interval_chunks_split_high_interval_count(tmp_path: Path) -> None:
    src = tmp_path / 'key_intervals.parquet'
    dest = tmp_path / 'interval_chunks.parquet'
    _interval_rows(250).write_parquet(src)
    ids = write_interval_chunks(src, dest, max_intervals=40)
    counts = pl.scan_parquet(dest).group_by('chunk_id').agg(n=pl.len()).collect()
    assert INDEXED_MAX_INTERVALS_PER_CHUNK == 50_000
    assert INDEXED_MAX_CHUNK_COST == 1_500_000_000
    assert counts['n'].max() <= 40
    assert 7 <= counts.height <= 16
    assert tuple(ids) == tuple(range(counts.height))
    assert int(counts.filter(pl.col('n') >= 1_000_000).select(pl.len()).item()) == 0

    n_wide = 3707
    band = 5_740_000
    width = 9_928
    stride = band // n_wide
    wide_src = tmp_path / 'wide_intervals.parquet'
    wide_dest = tmp_path / 'wide_chunks.parquet'
    pl.DataFrame({
        'color': list(range(n_wide)),
        'key': [f'k{i}' for i in range(n_wide)],
        'left': [i * stride for i in range(n_wide)],
        'right': [i * stride + width for i in range(n_wide)],
    }).write_parquet(wide_src)
    assert n_wide < INDEXED_MAX_INTERVALS_PER_CHUNK
    assert n_wide * band > INDEXED_MAX_CHUNK_COST
    wide_ids = write_interval_chunks(wide_src, wide_dest)
    wide = (
        pl
        .scan_parquet(wide_src)
        .join(pl.scan_parquet(wide_dest), on='color')
        .group_by('chunk_id')
        .agg(n=pl.len(), band=pl.col('right').max() - pl.col('left').min())
        .with_columns(cost=pl.col('n') * pl.col('band'))
        .collect()
    )
    assert len(wide_ids) > 1
    assert wide['n'].max() < n_wide
    assert wide['cost'].max() <= INDEXED_MAX_CHUNK_COST


def test_interval_chunks_dense_leftover_not_quantum_oversplit(tmp_path: Path) -> None:
    """A long left span must not emit one chunk per cost/interval quantum."""
    n = 8_000
    span = 810_000_000
    stride = span // n
    quantum = INDEXED_MAX_CHUNK_COST // INDEXED_MAX_INTERVALS_PER_CHUNK
    assert span // quantum > 20_000
    src = tmp_path / 'dense_leftover.parquet'
    dest = tmp_path / 'dense_chunks.parquet'
    pl.DataFrame({
        'color': list(range(n)),
        'key': [f'k{i}' for i in range(n)],
        'left': [i * stride for i in range(n)],
        'right': [i * stride + 10 for i in range(n)],
    }).write_parquet(src)
    ids = write_interval_chunks(src, dest)
    costs = (
        pl
        .scan_parquet(src)
        .join(pl.scan_parquet(dest), on='color')
        .group_by('chunk_id')
        .agg(n=pl.len(), band=pl.col('right').max() - pl.col('left').min())
        .with_columns(cost=pl.col('n') * pl.col('band'))
        .collect()
    )
    assert len(ids) < 2_000
    assert len(ids) < span // quantum // 10
    assert costs['n'].min() > 1
    assert costs['cost'].max() <= INDEXED_MAX_CHUNK_COST


def test_interval_chunks_resume_keeps_finished_ids(tmp_path: Path) -> None:
    src = tmp_path / 'key_intervals.parquet'
    dest = tmp_path / 'interval_chunks.parquet'
    work = tmp_path / 'work'
    work.mkdir()
    _interval_rows(80).write_parquet(src)
    pl.DataFrame({'color': [0], 'key': ['k'], 'left': [0], 'right': [1]}).write_parquet(
        work / 'key_intervals.parquet'
    )
    pl.DataFrame(
        {'rank': [0], 'parent_color': [0]}, schema={'rank': pl.Int64, 'parent_color': pl.Int32}
    ).write_parquet(work / 'sa_color.parquet')
    pl.DataFrame({
        'color': [0],
        'term': ['t'],
        'key': ['k'],
        'norm_nbytes': [1],
        'tier': ['short'],
    }).write_parquet(work / 'term_identity.parquet')
    first = write_interval_chunks(src, dest, max_intervals=40)
    assert first == (0, 1)
    kept = (
        pl.scan_parquet(dest).filter(pl.col('chunk_id') == 0).select('color', 'chunk_id').collect()
    )
    store = ScoreStageStore(tmp_path)
    assert INDEXED_CONTRIB_CHUNKS == 256
    stamp = IndexedMapsFingerprint.from_maps(work, INDEXED_CONTRIB_CHUNKS)
    chunk_dir = store.bind_indexed_contrib_chunks(work, stamp)
    pl.DataFrame(
        {'child': ['a'], 'p_ta': [1], 'sum_parent_tf': [1.0]},
        schema={'child': pl.String, 'p_ta': pl.Int64, 'sum_parent_tf': pl.Float64},
    ).write_parquet(chunk_dir / '00000.parquet')
    finished_mtime = (chunk_dir / '00000.parquet').stat().st_mtime_ns
    second = write_interval_chunks(
        src,
        dest,
        finished_ids=(0,),
        max_intervals=10,
    )
    rebound = store.bind_indexed_contrib_chunks(work, stamp)
    assigned = pl.scan_parquet(dest).collect()
    kept_after = assigned.filter(pl.col('chunk_id') == 0).select('color', 'chunk_id').sort('color')
    pending = assigned.filter(pl.col('chunk_id') != 0)
    assert rebound == chunk_dir
    assert kept_after.equals(kept.sort('color'))
    assert pending.group_by('chunk_id').agg(n=pl.len())['n'].max() <= 10
    assert 0 in second
    assert max(second) >= 4
    assert (chunk_dir / '00000.parquet').is_file()
    assert (chunk_dir / '00000.parquet').stat().st_mtime_ns == finished_mtime


def test_later_stages_log_progress_on_tiny_fixture(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', _mixed_rows())
    dest = tmp_path / 'indexed'
    with capture_logs() as logs:
        root = score_term_parquet(
            extract,
            ate=_indexed_ate(),
            temp_dir=tmp_path / 'spill',
            artifact_dir=dest,
        )
    stages = {entry.get('stage') for entry in logs}
    assert 'parent_contrib_merge' in stages
    assert 'scored' in stages
    assert 'scored_chunk' in stages
    assert 'keys' in stages
    assert (dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet').is_file()
    assert (dest / SCORE_STAGE_DIRNAME / 'scored.parquet').is_file()
    assert (dest / SCORE_STAGE_DIRNAME / SCORED_CHUNK_DIRNAME).is_dir()
    assert TermhoodStore.open(root).meta.n_keys > 0

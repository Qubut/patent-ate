"""Staged C-value algebra, artifacts, resume, and publication."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import ibis
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jate import CValue
from jate.features import Containment, TermComponentIndex, TermFrequency
from jate.models import Candidate

from patent_ate.cvalue import ParentScorePlan, plan_term_score, score_term_parquet
from patent_ate.cvalue.algebra import (
    candidate_parent_contributions,
    candidate_span_census,
    candidate_span_lengths,
    cvalue_keys,
    parent_bucket,
    parent_contributions,
    parent_span_costs,
    parent_windows,
    scored_terms,
    surfaces,
    term_stats,
)
from patent_ate.cvalue.exec import ScoreStageExecutor
from patent_ate.cvalue.plan import (
    PARENT_PLAN_NAME,
    PARENT_SPAN_COSTS_NAME,
    PARENT_UNIT_DIRNAME,
    HashSlotUnit,
    ParentWorkPlan,
    ParentWorkPlanError,
    SpanBandUnit,
    build_parent_work_plan,
)
from patent_ate.cvalue.store import (
    CANDIDATE_SPAN_LENGTHS_NAME,
    SCORE_MANIFEST_NAME,
    SCORE_SEMANTIC_VERSION,
    SCORE_STAGE_DIRNAME,
    STAGE_SCHEMAS,
    ExtractFingerprint,
    ScoreStageManifest,
    ScoreStageStore,
)
from patent_ate.cvalue.text import (
    JATE_CHUNK_PATTERN,
    JATE_WORD_CLASS,
    jate_contains,
    jate_word_count,
)
from patent_ate.nlp import JateDraw
from patent_ate.plan import PATENT_ID_COLUMN
from patent_ate.spec import AteSpec
from patent_ate.termhood import (
    TERMHOOD_META_NAME,
    TERMHOOD_PARQUET_NAME,
    TermhoodStore,
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


def _write_compact(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_COMPACT_SCHEMA), path)
    return path


def _coil_rows() -> list[dict[str, Any]]:
    return [
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


def _patents(rows: list[dict[str, Any]]) -> ibis.Table:
    return ibis.memtable(pa.Table.from_pylist(rows, schema=_COMPACT_SCHEMA))


def _sorted_frame(frame: pl.DataFrame, key: str) -> pl.DataFrame:
    return frame.sort(key).select(sorted(frame.columns))


def _score_ate(**overrides: Any) -> AteSpec:
    values = {'duckdb_memory': '256MB', 'duckdb_threads': 2, 'parent_buckets': 1}
    values.update(overrides)
    return AteSpec(**values)


_STALE_STAGE = b'stale-stage'


def _poison_stage(path: Path) -> None:
    path.write_bytes(_STALE_STAGE)


def _assert_parquet_replaced(path: Path) -> None:
    assert path.is_file()
    assert path.read_bytes() != _STALE_STAGE
    _ = pl.scan_parquet(path).collect()


def _assert_bucket_sql(sql: str, *, buckets: int, bucket: int) -> None:
    upper = sql.upper()
    assert 'HASH(' in upper
    assert '%' in sql
    assert str(buckets) in sql
    assert re.search(rf'(=|= )\s*{bucket}\b', sql) is not None
    assert 'GROUP BY' in upper
    assert 'RANGE' in upper
    assert 'REGEXP_EXTRACT_ALL' in upper
    assert 'SPAN_N' in upper
    range_at = upper.index('RANGE')
    span_at = upper.index('SPAN_N')
    assert span_at < range_at


def _merged_contrib(stats: ibis.Table, buckets: int) -> pl.DataFrame:
    parts = [
        parent_contributions(stats, buckets=buckets, bucket=index).to_polars()
        for index in range(buckets)
    ]
    nonempty = [part for part in parts if part.height > 0]
    if not nonempty:
        return parts[0]
    return (
        pl
        .concat(nonempty)
        .group_by('child')
        .agg(pl.col('p_ta').sum(), pl.col('sum_parent_tf').sum())
    )


def test_repeated_child_in_one_parent_counts_once() -> None:
    rows = [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': [
                {'term': 'a a a', 'frequency': 4, 'surfaces': ['a a a']},
                {'term': 'a a', 'frequency': 2, 'surfaces': ['a a']},
            ],
        }
    ]
    contrib = parent_contributions(term_stats(_patents(rows))).to_polars()
    child = contrib.filter(pl.col('child') == 'a a')
    assert child.height == 1
    assert int(child['p_ta'][0]) == 1
    assert float(child['sum_parent_tf'][0]) == pytest.approx(4.0)


def test_punctuation_boundaries_match_original_string_windows() -> None:
    rows = [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': [
                {
                    'term': 'anti-lock brake system',
                    'frequency': 1,
                    'surfaces': ['anti-lock brake system'],
                },
                {'term': 'lock brake', 'frequency': 2, 'surfaces': ['lock brake']},
                {'term': 'anti lock', 'frequency': 1, 'surfaces': ['anti lock']},
                {'term': 'foo/bar extra', 'frequency': 2, 'surfaces': ['foo/bar extra']},
                {
                    'term': 'pre-foo-bar extra pre-foo/bar extra',
                    'frequency': 1,
                    'surfaces': ['pre-foo-bar extra pre-foo/bar extra'],
                },
            ],
        }
    ]
    scored = parent_contributions(term_stats(_patents(rows))).to_polars()
    contrib = {str(row['child']): int(row['p_ta']) for row in scored.iter_rows(named=True)}
    assert contrib['lock brake'] == 1
    assert 'anti lock' not in contrib
    assert contrib['foo/bar extra'] == 1


def test_underscore_apostrophe_and_digit_glued_forms() -> None:
    rows = [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': [
                {'term': 'foo_bar extra word', 'frequency': 1, 'surfaces': ['foo_bar extra word']},
                {'term': 'foo_bar extra', 'frequency': 2, 'surfaces': ['foo_bar extra']},
                {'term': 'foo', 'frequency': 3, 'surfaces': ['foo']},
                {'term': "don't stop extra", 'frequency': 1, 'surfaces': ["don't stop extra"]},
                {'term': "don't stop", 'frequency': 2, 'surfaces': ["don't stop"]},
                {'term': 'stop extra', 'frequency': 1, 'surfaces': ['stop extra']},
                {'term': 'iso9001 extra word', 'frequency': 1, 'surfaces': ['iso9001 extra word']},
                {'term': 'iso9001 extra', 'frequency': 2, 'surfaces': ['iso9001 extra']},
                {'term': 'iso', 'frequency': 3, 'surfaces': ['iso']},
                {'term': '9001 extra', 'frequency': 1, 'surfaces': ['9001 extra']},
            ],
        }
    ]
    stats = term_stats(_patents(rows))
    contrib = parent_contributions(stats).to_polars()
    children = set(contrib['child'].to_list())
    assert 'foo_bar extra' in children
    assert 'foo' not in children
    assert "don't stop" in children
    assert 'stop extra' in children
    assert 'iso9001 extra' in children
    assert 'iso' not in children
    assert '9001 extra' not in children
    merged = _merged_contrib(stats, 4)
    assert _sorted_frame(merged, 'child').equals(_sorted_frame(contrib, 'child'))


def test_per_parent_list_distinct_matches_global_distinct() -> None:
    stats = term_stats(
        _patents([
            *_coil_rows(),
            {
                PATENT_ID_COLUMN: 'p3',
                'n_docs': 1,
                'terms': [
                    {'term': 'a a a', 'frequency': 3, 'surfaces': ['a a a']},
                    {'term': 'a a', 'frequency': 1, 'surfaces': ['a a']},
                ],
            },
        ])
    )
    windows = parent_windows(stats.filter(stats.word_count >= 2))
    doubled = windows.union(windows)
    global_rows = (
        doubled
        .distinct()
        .group_by('window')
        .agg(p_ta=ibis._.count(), sum_parent_tf=ibis._.parent_tf.sum())
        .to_polars()
    )
    local_rows = (
        doubled
        .group_by('parent', 'parent_tf', 'parent_word_count')
        .agg(window=ibis._.window.collect())
        .mutate(window=ibis._.window.unique())
        .unnest('window')
        .group_by('window')
        .agg(p_ta=ibis._.count(), sum_parent_tf=ibis._.parent_tf.sum())
        .to_polars()
    )
    assert _sorted_frame(local_rows, 'window').equals(_sorted_frame(global_rows, 'window'))
    unique_sql = ibis.to_sql(
        ibis.memtable({'xs': [['a', 'a', 'b']]}).select(
            u=ibis._.xs.unique(),
        )
    )
    assert 'LIST_DISTINCT' in unique_sql.upper() or 'ARRAY_DISTINCT' in unique_sql.upper()


def test_contribution_before_join_matches_join_then_aggregate() -> None:
    stats = term_stats(_patents(_coil_rows()))
    windows = parent_windows(stats.filter(stats.word_count >= 2))
    proper = windows.filter(windows.parent_word_count > jate_word_count(windows.window))
    before = (
        proper
        .group_by(window=proper.window)
        .agg(p_ta=ibis._.count(), sum_parent_tf=proper.parent_tf.sum())
        .inner_join(stats, proper.window == stats.term)
        .select(child=stats.term, p_ta=ibis._.p_ta, sum_parent_tf=ibis._.sum_parent_tf)
        .to_polars()
    )
    after = (
        proper
        .inner_join(stats, proper.window == stats.term)
        .group_by(child=stats.term)
        .agg(p_ta=ibis._.count(), sum_parent_tf=proper.parent_tf.sum())
        .to_polars()
    )
    got = parent_contributions(stats).to_polars()
    assert _sorted_frame(before, 'child').equals(_sorted_frame(after, 'child'))
    assert _sorted_frame(got, 'child').equals(_sorted_frame(before, 'child'))


def test_stage_sql_requires_bucket_predicate_and_compact_aggregate() -> None:
    patents = _patents(_coil_rows())
    stats_sql = ibis.to_sql(term_stats(patents))
    surfaces_sql = ibis.to_sql(surfaces(patents))
    compact = ibis.memtable({
        'term': ['coil spring', 'coil spring assembly', 'vehicle interior'],
        'tf': [2, 1, 1],
        'df': [1, 1, 1],
        'word_count': [2, 3, 2],
    })
    one_sql = ibis.to_sql(parent_contributions(compact, buckets=1, bucket=0))
    bucket_sql = ibis.to_sql(parent_contributions(compact, buckets=4, bucket=1))
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
    assert 'RANGE' not in stats_sql.upper()
    assert 'RANGE' not in surfaces_sql.upper()
    assert 'REGEXP_EXTRACT_ALL' not in surfaces_sql.upper()
    assert 'surfaces' not in one_sql.lower()
    assert 'surfaces' not in bucket_sql.lower()
    _assert_bucket_sql(one_sql, buckets=1, bucket=0)
    _assert_bucket_sql(bucket_sql, buckets=4, bucket=1)
    assert 'RANGE' not in scored_sql.upper()
    assert 'UNNEST' not in scored_sql
    assert 'REGEXP_EXTRACT_ALL' not in scored_sql.upper()
    assert 'RANGE' not in keys_sql.upper()
    assert 'UNNEST' not in keys_sql
    assert 'REGEXP_EXTRACT_ALL' not in keys_sql.upper()
    assigned = compact.mutate(bucket=parent_bucket(compact.term, 4)).to_polars()
    parents = assigned.filter(pl.col('word_count') >= 2)
    for index in range(4):
        subset = set(parents.filter(pl.col('bucket') == index)['term'].to_list())
        assert subset < set(parents['term'].to_list()) or parents.height <= 1
    assert set(parents['bucket'].to_list()) <= {0, 1, 2, 3}


def test_score_stages_write_declared_schemas_and_publish(
    tmp_path: Path,
) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    leftover = dest / f'{TERMHOOD_PARQUET_NAME}.partial'
    dest.mkdir()
    _ = leftover.write_text('stale', encoding='utf-8')
    spill = dest / 'duckdb_tmp'
    spill.mkdir()
    _ = (spill / 'old.bin').write_bytes(b'stale')
    path = score_term_parquet(
        extract,
        ate=_score_ate(),
        temp_dir=spill,
        artifact_dir=dest,
    )
    assert path == dest
    store = TermhoodStore.open(dest)
    assert store.meta.total_docs == 2
    assert store.meta.n_keys >= 1
    assert not leftover.exists()
    assert not (dest / f'{TERMHOOD_PARQUET_NAME}.partial').exists()
    assert not (dest / f'{TERMHOOD_META_NAME}.partial').exists()
    assert not any(path.is_file() for path in spill.rglob('*'))
    stages = dest / SCORE_STAGE_DIRNAME
    for name, schema in STAGE_SCHEMAS.items():
        frame = pl.scan_parquet(stages / f'{name}.parquet').collect()
        assert frame.schema == schema
        assert frame.height > 0
        assert not (stages / f'{name}.parquet.partial').exists()
    manifest = ScoreStageManifest.model_validate_json(
        (stages / SCORE_MANIFEST_NAME).read_text(encoding='utf-8')
    )
    assert manifest.semantic_version == SCORE_SEMANTIC_VERSION
    assert set(manifest.completed) >= {
        'term_stats',
        'surfaces',
        CANDIDATE_SPAN_LENGTHS_NAME,
        PARENT_SPAN_COSTS_NAME,
        'parent_contrib',
        'scored',
    }
    assert manifest.parent_buckets == 1
    assert manifest.tail_window_cap == 400_000
    assert manifest.tail_weighted_bytes_cap == 400_000_000
    assert manifest.base_window_cap == 25_000_000
    assert manifest.completed_units
    assert manifest.extract == ExtractFingerprint.from_files((extract / 'part-0.parquet',))
    assert (stages / PARENT_PLAN_NAME).is_file()
    assert (stages / PARENT_UNIT_DIRNAME).is_dir()
    plan = ParentWorkPlan.model_validate_json(
        (stages / PARENT_PLAN_NAME).read_text(encoding='utf-8')
    )
    assert set(manifest.completed_units) == set(plan.unit_ids())
    for unit in plan.units:
        if isinstance(unit, HashSlotUnit):
            assert unit.total_slots == 1
            assert unit.estimated_windows <= 25_000_000
        else:
            assert unit.estimated_windows <= 400_000
            assert unit.estimated_weighted_bytes <= 400_000_000
            assert unit.strategy in {'window', 'candidate'}
        assert (stages / PARENT_UNIT_DIRNAME / f'{unit.unit_id}.parquet').is_file()


def test_store_and_executor_public_contracts(tmp_path: Path) -> None:
    ate = _score_ate()
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    _write_compact(extract / 'termhood.parquet', _coil_rows())
    files = ScoreStageStore.extract_files(extract)
    assert {path.name for path in files} == {'part-0.parquet'}
    fingerprint = ExtractFingerprint.from_files(files)
    manifest = ScoreStageManifest.from_ate(extract=fingerprint, ate=ate, total_docs=2)
    assert manifest.semantic_version == SCORE_SEMANTIC_VERSION
    assert manifest.public_prefix_matches(fingerprint, ate)
    assert manifest.unit_config_matches(ate)
    store = ScoreStageStore(dest)
    assert not store.parquet_valid(store.stage_parquet('term_stats'), STAGE_SCHEMAS['term_stats'])
    session = store.bind(extract, ate)
    executor = ScoreStageExecutor(ate, dest / 'spill')
    executor.materialize_term_stats(session)
    executor.materialize_stage(
        session,
        CANDIDATE_SPAN_LENGTHS_NAME,
        lambda connection: candidate_span_lengths(
            connection.read_parquet(session.store.stage_parquet('term_stats'))
        ),
    )
    assert store.stage_valid('term_stats')
    filt = f'{PARENT_UNIT_DIRNAME}/filters/span-0001.parquet'
    store.write_filters({filt: ('coil spring assembly',)}, resume_units=False)
    unit = SpanBandUnit(
        unit_id='span-0001',
        filter_artifact=filt,
        span_n_min=1,
        span_n_max=8,
        estimated_windows=1,
        estimated_weighted_bytes=1,
        estimated_candidate_rows=1,
        estimated_candidate_bytes=1,
        estimated_comparisons=1,
        strategy='candidate',
    )
    plan = ParentWorkPlan(
        parent_buckets=ate.parent_buckets,
        tail_window_cap=ate.tail_window_cap,
        tail_weighted_bytes_cap=ate.tail_weighted_bytes_cap,
        base_window_cap=ate.base_window_cap,
        base_weighted_bytes_cap=ate.base_weighted_bytes_cap,
        max_parent_units=ate.max_parent_units,
        max_candidate_scans=ate.max_candidate_scans,
        tail_candidate_row_cap=ate.tail_candidate_row_cap,
        tail_compare_cap=ate.tail_compare_cap,
        full_candidate_scans=0,
        band_filtered_scans=1,
        units=(unit,),
    )

    def compiled_sql(connection: object) -> str:
        return ibis.to_sql(
            executor.contributions_for(connection, unit, plan, session)  # type: ignore[arg-type]
        )

    sql = executor.run(compiled_sql)
    assert 'RANGE' not in sql.upper()
    assert 'REGEXP_ESCAPE' in sql.upper()
    assert JATE_WORD_CLASS in sql


def test_resume_skips_valid_term_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    ate = _score_ate()
    spill = dest / 'duckdb_tmp'
    calls: list[str] = []
    real = term_stats

    def spy(patents: ibis.Table) -> ibis.Table:
        calls.append('term_stats')
        return real(patents)

    monkeypatch.setattr('patent_ate.cvalue.algebra.term_stats', spy)
    first = score_term_parquet(extract, ate=ate, temp_dir=spill, artifact_dir=dest)
    assert calls == ['term_stats']
    stats_path = dest / SCORE_STAGE_DIRNAME / 'term_stats.parquet'
    before = stats_path.stat()
    second = score_term_parquet(extract, ate=ate, temp_dir=spill, artifact_dir=dest)
    after = stats_path.stat()
    assert first == second == dest
    assert calls == ['term_stats']
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_size == before.st_size


def test_partial_and_spill_cleanup(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    stages = dest / SCORE_STAGE_DIRNAME
    stages.mkdir(parents=True)
    stale = stages / 'term_stats.parquet.partial'
    _ = stale.write_text('stale', encoding='utf-8')
    spill = dest / 'duckdb_tmp'
    spill.mkdir()
    _ = (spill / 'leftover.tmp').write_bytes(b'x')
    score_term_parquet(
        extract,
        ate=_score_ate(),
        temp_dir=spill,
        artifact_dir=dest,
    )
    assert not stale.exists()
    assert not list(stages.glob('*.partial'))
    assert not any(path.is_file() for path in spill.rglob('*'))


def test_config_mismatch_invalidates_score_stages_only(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    part = _write_compact(extract / 'part-0.parquet', _coil_rows())
    extract_bytes = part.read_bytes()
    spill = dest / 'duckdb_tmp'
    score_term_parquet(
        extract,
        ate=_score_ate(),
        temp_dir=spill,
        artifact_dir=dest,
    )
    stats_path = dest / SCORE_STAGE_DIRNAME / 'term_stats.parquet'
    _poison_stage(stats_path)
    score_term_parquet(
        extract,
        ate=_score_ate(duckdb_memory='128MB'),
        temp_dir=spill,
        artifact_dir=dest,
    )
    assert part.read_bytes() == extract_bytes
    _assert_parquet_replaced(stats_path)
    manifest = ScoreStageManifest.model_validate_json(
        (dest / SCORE_STAGE_DIRNAME / SCORE_MANIFEST_NAME).read_text(encoding='utf-8')
    )
    assert manifest.duckdb_memory == '128MB'
    assert manifest.extract.sum_bytes == len(extract_bytes)


def test_extract_fingerprint_mismatch_invalidates_score_stages_only(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    part = _write_compact(extract / 'part-0.parquet', _coil_rows())
    extract_bytes = part.read_bytes()
    spill = dest / 'duckdb_tmp'
    score_term_parquet(
        extract,
        ate=_score_ate(),
        temp_dir=spill,
        artifact_dir=dest,
    )
    manifest_path = dest / SCORE_STAGE_DIRNAME / SCORE_MANIFEST_NAME
    loaded = ScoreStageManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    _ = manifest_path.write_text(
        loaded.model_copy(
            update={
                'extract': loaded.extract.model_copy(update={'digest': '0' * 64}),
            }
        ).model_dump_json()
        + '\n',
        encoding='utf-8',
    )
    marker = dest / SCORE_STAGE_DIRNAME / 'stale.marker'
    _ = marker.write_text('drop', encoding='utf-8')
    stats_path = dest / SCORE_STAGE_DIRNAME / 'term_stats.parquet'
    _poison_stage(stats_path)
    score_term_parquet(
        extract,
        ate=_score_ate(),
        temp_dir=spill,
        artifact_dir=dest,
    )
    assert part.read_bytes() == extract_bytes
    assert not marker.exists()
    _assert_parquet_replaced(stats_path)
    restored = ScoreStageManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    assert restored.extract.digest != '0' * 64
    assert restored.extract == ExtractFingerprint.from_files((part,))


def test_hash_range_buckets_match_unbucketed_contributions() -> None:
    rows = _coil_rows() + [
        {
            PATENT_ID_COLUMN: f'p{index}',
            'n_docs': 1,
            'terms': [
                {
                    'term': f'term{index:02d} extra word',
                    'frequency': 1,
                    'surfaces': [f'term{index:02d} extra word'],
                },
                {
                    'term': f'term{index:02d} extra',
                    'frequency': 2,
                    'surfaces': [f'term{index:02d} extra'],
                },
            ],
        }
        for index in range(3, 19)
    ]
    stats = term_stats(_patents(rows))
    unbucketed = parent_contributions(stats).to_polars()
    one = parent_contributions(stats, buckets=1, bucket=0).to_polars()
    merged = _merged_contrib(stats, 4)
    assert _sorted_frame(one, 'child').equals(_sorted_frame(unbucketed, 'child'))
    assert _sorted_frame(merged, 'child').equals(_sorted_frame(unbucketed, 'child'))
    assigned = stats.filter(stats.word_count >= 2).mutate(
        bucket=parent_bucket(ibis._.term, 4),
    )
    parents = assigned.to_polars()
    all_parents = set(parents['term'].to_list())
    seen: set[str] = set()
    for index in range(4):
        subset = set(parents.filter(pl.col('bucket') == index)['term'].to_list())
        assert subset <= all_parents
        assert subset != all_parents
        seen.update(subset)
    assert seen == all_parents


def test_score_buckets_match_unbucketed_and_jate(tmp_path: Path) -> None:
    def jate_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
        pool: dict[str, Candidate] = {}
        for row in rows:
            for item in row['terms']:
                term = str(item['term'])
                cand = pool.setdefault(term, Candidate(surface_form=term, normalized_form=term))
                cand.surface_forms.update(item['surfaces'])
                for index in range(int(item['frequency'])):
                    cand.add_position(str(row[PATENT_ID_COLUMN]), index, index + 1, 0)
        candidates = list(pool.values())
        docs = sum(int(row['n_docs']) for row in rows)
        freq = TermFrequency.build(candidates, docs)
        ranked = (
            CValue().score(candidates, freq).filter_by_frequency(1).filter_by_length(min_words=2)
        )
        return JateDraw.table_from_cvalue(ranked, freq).c_values

    extract = tmp_path / 'extract'
    dest_one = tmp_path / 'one'
    dest_many = tmp_path / 'many'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    one = TermhoodStore.open(
        score_term_parquet(
            extract,
            ate=_score_ate(),
            temp_dir=tmp_path / 'spill_one',
            artifact_dir=dest_one,
        )
    )
    many = TermhoodStore.open(
        score_term_parquet(
            extract,
            ate=_score_ate(parent_buckets=4),
            temp_dir=tmp_path / 'spill_many',
            artifact_dir=dest_many,
        )
    )
    one_frame = pl.scan_parquet(one.parquet).collect().sort('key')
    many_frame = pl.scan_parquet(many.parquet).collect().sort('key')
    assert one_frame.equals(many_frame)
    expected = jate_scores(_coil_rows())
    scored = dict(one_frame.select('key', 'c_value').iter_rows())
    assert scored.keys() == expected.keys()
    for key, value in expected.items():
        assert scored[key] == pytest.approx(value)
    assert one.meta.total_docs == many.meta.total_docs == 2
    manifest = ScoreStageManifest.model_validate_json(
        (dest_many / SCORE_STAGE_DIRNAME / SCORE_MANIFEST_NAME).read_text(encoding='utf-8')
    )
    assert manifest.parent_buckets == 4
    assert manifest.completed_units
    plan = ParentWorkPlan.model_validate_json(
        (dest_many / SCORE_STAGE_DIRNAME / PARENT_PLAN_NAME).read_text(encoding='utf-8')
    )
    assert set(manifest.completed_units) == set(plan.unit_ids())


def test_unigram_only_zero_row_contrib_is_resumable(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(
        extract / 'part-0.parquet',
        [
            {
                PATENT_ID_COLUMN: 'p1',
                'n_docs': 1,
                'terms': [
                    {'term': 'coil', 'frequency': 2, 'surfaces': ['coil']},
                    {'term': 'spring', 'frequency': 1, 'surfaces': ['Spring']},
                ],
            }
        ],
    )
    ate = _score_ate()
    first = score_term_parquet(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    store = TermhoodStore.open(first)
    assert store.meta.total_docs == 1
    assert store.meta.n_keys == 0
    contrib = dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet'
    scored = dest / SCORE_STAGE_DIRNAME / 'scored.parquet'
    assert contrib.stat().st_size > 0
    assert scored.stat().st_size > 0
    assert pl.scan_parquet(contrib).collect().height == 0
    assert pl.scan_parquet(scored).collect().height == 0
    before = contrib.stat()
    score_term_parquet(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    after = contrib.stat()
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns


def test_missing_unit_recomputes_only_that_unit(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    ate = _score_ate(parent_buckets=4)
    score_term_parquet(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    stages = dest / SCORE_STAGE_DIRNAME
    plan = ParentWorkPlan.model_validate_json(
        (stages / PARENT_PLAN_NAME).read_text(encoding='utf-8')
    )
    unit_dir = stages / PARENT_UNIT_DIRNAME
    kept = {unit.unit_id: (unit_dir / f'{unit.unit_id}.parquet').stat() for unit in plan.units}
    missing = plan.units[min(2, len(plan.units) - 1)].unit_id
    (unit_dir / f'{missing}.parquet').unlink()
    (stages / 'parent_contrib.parquet').unlink()
    manifest_path = stages / SCORE_MANIFEST_NAME
    loaded = ScoreStageManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    completed = tuple(name for name in loaded.completed if name != 'parent_contrib')
    _ = manifest_path.write_text(
        loaded.model_copy(update={'completed': completed}).model_dump_json() + '\n',
        encoding='utf-8',
    )
    score_term_parquet(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    assert (unit_dir / f'{missing}.parquet').is_file()
    assert all(
        (unit_dir / f'{unit_id}.parquet').stat().st_ino == before.st_ino
        and (unit_dir / f'{unit_id}.parquet').stat().st_mtime_ns == before.st_mtime_ns
        for unit_id, before in kept.items()
        if unit_id != missing
    )
    store = TermhoodStore.open(dest)
    assert store.meta.total_docs == 2
    assert store.meta.n_keys >= 1


def test_parent_bucket_count_change_invalidates_score_stages_only(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    part = _write_compact(extract / 'part-0.parquet', _coil_rows())
    extract_bytes = part.read_bytes()
    score_term_parquet(
        extract,
        ate=_score_ate(),
        temp_dir=dest / 'spill',
        artifact_dir=dest,
    )
    stats_path = dest / SCORE_STAGE_DIRNAME / 'term_stats.parquet'
    first_ino = stats_path.stat().st_ino
    score_term_parquet(
        extract,
        ate=_score_ate(parent_buckets=4),
        temp_dir=dest / 'spill',
        artifact_dir=dest,
    )
    assert part.read_bytes() == extract_bytes
    assert stats_path.stat().st_ino == first_ino
    manifest = ScoreStageManifest.model_validate_json(
        (dest / SCORE_STAGE_DIRNAME / SCORE_MANIFEST_NAME).read_text(encoding='utf-8')
    )
    assert manifest.parent_buckets == 4
    assert manifest.completed_units


def test_bounded_unit_profile_equal_outputs_and_fewer_windows(tmp_path: Path) -> None:
    rows = [
        {
            PATENT_ID_COLUMN: f'p{index:02d}',
            'n_docs': 1,
            'terms': [
                {
                    'term': f'alpha{index:02d} beta gamma delta',
                    'frequency': 1,
                    'surfaces': [f'alpha{index:02d} beta gamma delta'],
                },
                {
                    'term': f'alpha{index:02d} beta gamma',
                    'frequency': 2,
                    'surfaces': [f'alpha{index:02d} beta gamma'],
                },
                {
                    'term': f'alpha{index:02d} beta',
                    'frequency': 3,
                    'surfaces': [f'alpha{index:02d} beta'],
                },
            ],
        }
        for index in range(48)
    ]
    extract = tmp_path / 'extract'
    _write_compact(extract / 'part-0.parquet', rows)
    stats = term_stats(_patents(rows))
    lengths = candidate_span_lengths(stats)
    windows = int(parent_windows(stats, lengths).count().to_pyarrow().as_py())
    triangular = 48 * (4 * 5 // 2 - 1 + 3 * 4 // 2 - 1 + 2 * 3 // 2 - 1)
    assert windows < triangular
    profiles: dict[int, dict[str, float]] = {}
    frames: dict[int, pl.DataFrame] = {}
    for buckets in (1, 32):
        dest = tmp_path / f'run_{buckets}'
        started = time.perf_counter()
        store = TermhoodStore.open(
            score_term_parquet(
                extract,
                ate=_score_ate(parent_buckets=buckets),
                temp_dir=dest / 'spill',
                artifact_dir=dest,
            )
        )
        wall = time.perf_counter() - started
        contrib = dest / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet'
        unit_dir = dest / SCORE_STAGE_DIRNAME / PARENT_UNIT_DIRNAME
        unit_bytes = sum(
            path.stat().st_size for path in unit_dir.glob('*.parquet') if path.parent == unit_dir
        )
        frames[buckets] = pl.scan_parquet(store.parquet).collect().sort('key')
        profiles[buckets] = {
            'wall_s': wall,
            'output_rows': float(frames[buckets].height),
            'contrib_bytes': float(contrib.stat().st_size),
            'unit_bytes': float(unit_bytes),
            'spill_bytes': 0.0,
        }
        assert not any(path.is_file() for path in (dest / 'spill').rglob('*'))
    assert frames[1].equals(frames[32])
    assert profiles[1]['output_rows'] == profiles[32]['output_rows']
    assert profiles[1]['wall_s'] > 0
    assert profiles[32]['wall_s'] > 0
    assert profiles[1]['contrib_bytes'] > 0
    assert profiles[32]['unit_bytes'] > 0


def test_candidate_span_lengths_use_chunker_not_whitespace() -> None:
    stats = ibis.memtable({
        'term': ['anti-lock brake', 'lock brake', 'only spaces'],
        'tf': [1, 1, 1],
        'df': [1, 1, 1],
        'word_count': [2, 2, 2],
    })
    lengths = set(candidate_span_lengths(stats).to_polars()['span_n'].to_list())
    assert 5 in lengths
    assert 3 in lengths
    assert 2 not in lengths


def test_parent_windows_omit_nonexistent_span_lengths() -> None:
    stats = ibis.memtable({
        'term': ['alpha beta gamma delta', 'alpha beta'],
        'tf': [1, 2],
        'df': [1, 1],
        'word_count': [4, 2],
    })
    lengths = candidate_span_lengths(stats)
    windows = parent_windows(stats, lengths).to_polars()
    spans = {
        str(row['window']): len(str(row['window']).split()) for row in windows.iter_rows(named=True)
    }
    assert 3 not in set(spans.values())
    assert 'alpha beta gamma' not in spans
    assert 'alpha beta' in set(windows['window'].to_list())
    sql = ibis.to_sql(parent_windows(stats, lengths))
    assert 'RANGE' in sql.upper()
    assert sql.upper().index('SPAN_N') < sql.upper().index('RANGE')


def test_length_band_union_matches_unbounded_and_does_not_double_count() -> None:
    stats = term_stats(_patents(_coil_rows()))
    lengths = candidate_span_lengths(stats)
    whole = parent_contributions(stats, lengths).to_polars()
    span_values = tuple(sorted(int(value) for value in lengths.to_polars()['span_n'].to_list()))
    ate = _score_ate(tail_window_cap=8, tail_weighted_bytes_cap=400_000_000)
    plan, filters = build_parent_work_plan(
        parent_span_costs(stats, lengths, buckets=ate.parent_buckets).to_polars(),
        span_values,
        ate=ate,
    )
    assert plan.units
    assert all(
        (
            unit.estimated_windows <= ate.base_window_cap
            if isinstance(unit, HashSlotUnit)
            else unit.estimated_windows <= ate.tail_window_cap
        )
        for unit in plan.units
    )
    excluded = set(filters.get(plan.over_cap_filter, ())) if plan.over_cap_filter else set()
    parts: list[pl.DataFrame] = []
    for unit in plan.units:
        if isinstance(unit, HashSlotUnit):
            parents = stats.filter(parent_bucket(stats.term, unit.total_slots) == unit.slot)
            if excluded:
                parents = parents.filter(~parents.term.isin(tuple(excluded)))
            parts.append(parent_contributions(stats, lengths, parents=parents).to_polars())
            continue
        term = filters[unit.filter_artifact][0]
        parts.append(
            parent_contributions(
                stats,
                lengths.filter(
                    (lengths.span_n >= unit.span_n_min) & (lengths.span_n <= unit.span_n_max)
                ),
                parents=stats.filter(stats.term == term),
            ).to_polars()
        )
    nonempty = [part for part in parts if part.height > 0]
    merged = (
        pl
        .concat(nonempty)
        .group_by('child')
        .agg(pl.col('p_ta').sum(), pl.col('sum_parent_tf').sum())
        if nonempty
        else whole
    )
    assert _sorted_frame(merged, 'child').equals(_sorted_frame(whole, 'child'))


def test_ten_thousand_word_preflight_splits_without_range() -> None:
    parent = ' '.join(f'w{index}' for index in range(10062))
    children = [
        ' '.join(f'c{length}_{index}' for index in range(length)) for length in range(2, 271)
    ]
    stats = ibis.memtable({
        'term': [parent, *children],
        'tf': [1] * (1 + len(children)),
        'df': [1] * (1 + len(children)),
        'word_count': [10062, *list(range(2, 271))],
    })
    lengths = candidate_span_lengths(stats)
    ate = _score_ate(parent_buckets=32)
    costs = parent_span_costs(stats, lengths, buckets=ate.parent_buckets)
    cost_sql = ibis.to_sql(costs)
    assert 'RANGE' not in cost_sql.upper()
    span_values = tuple(sorted(int(value) for value in lengths.to_polars()['span_n'].to_list()))
    assert len(span_values) >= 270
    cost_frame = costs.to_polars()
    plan, filters = build_parent_work_plan(cost_frame, span_values, ate=ate)
    assert all(
        (
            unit.estimated_windows <= ate.base_window_cap
            and unit.estimated_weighted_bytes <= ate.base_weighted_bytes_cap
            if isinstance(unit, HashSlotUnit)
            else unit.estimated_windows <= ate.tail_window_cap
            and unit.estimated_weighted_bytes <= ate.tail_weighted_bytes_cap
        )
        for unit in plan.units
    )
    band_units = [unit for unit in plan.units if isinstance(unit, SpanBandUnit)]
    assert band_units
    assert any(filters[unit.filter_artifact] == (parent,) for unit in band_units)
    replay, _ = build_parent_work_plan(cost_frame, span_values, ate=ate)
    assert replay == plan


def test_resume_semantic3_keeps_stats_and_surfaces(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    ate = _score_ate()
    score_term_parquet(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    stages = dest / SCORE_STAGE_DIRNAME
    stats_path = stages / 'term_stats.parquet'
    surfaces_path = stages / 'surfaces.parquet'
    stats_before = stats_path.stat()
    surfaces_before = surfaces_path.stat()
    stale_units = stages / 'parent_units_semantic3'
    stale_units.mkdir(exist_ok=True)
    stale = stale_units / 'hash-1024-0.parquet'
    _ = stale.write_bytes(stats_path.read_bytes())
    stale_ino = stale.stat().st_ino
    manifest_path = stages / SCORE_MANIFEST_NAME
    loaded = ScoreStageManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    _ = manifest_path.write_text(
        loaded.model_copy(
            update={
                'semantic_version': '3',
                'completed': ('term_stats', 'surfaces', 'parent_contrib', 'scored'),
                'completed_units': (),
            }
        ).model_dump_json()
        + '\n',
        encoding='utf-8',
    )
    (stages / 'parent_contrib.parquet').unlink()
    score_term_parquet(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    assert stats_path.stat().st_ino == stats_before.st_ino
    assert surfaces_path.stat().st_ino == surfaces_before.st_ino
    assert stale.stat().st_ino == stale_ino
    restored = ScoreStageManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    assert restored.semantic_version == SCORE_SEMANTIC_VERSION
    assert restored.completed_units
    assert (stages / 'parent_contrib.parquet').is_file()


def test_empty_unit_file_is_resumable_among_nonempty_siblings(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    extra = [
        {
            PATENT_ID_COLUMN: f'p{index:02d}',
            'n_docs': 1,
            'terms': [
                {
                    'term': f'term{index:02d} extra word',
                    'frequency': 1,
                    'surfaces': [f'term{index:02d} extra word'],
                },
                {
                    'term': f'term{index:02d} extra',
                    'frequency': 2,
                    'surfaces': [f'term{index:02d} extra'],
                },
            ],
        }
        for index in range(3, 19)
    ]
    _write_compact(extract / 'part-0.parquet', [*_coil_rows(), *extra])
    ate = _score_ate(parent_buckets=4)
    score_term_parquet(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    stages = dest / SCORE_STAGE_DIRNAME
    plan = ParentWorkPlan.model_validate_json(
        (stages / PARENT_PLAN_NAME).read_text(encoding='utf-8')
    )
    unit_dir = stages / PARENT_UNIT_DIRNAME
    assert len(plan.units) >= 2
    empty_id = plan.units[0].unit_id
    empty_path = unit_dir / f'{empty_id}.parquet'
    nonempty = [unit for unit in plan.units if unit.unit_id != empty_id]
    assert any(
        pl.scan_parquet(unit_dir / f'{unit.unit_id}.parquet').select(pl.len()).collect().item() > 0
        for unit in nonempty
    )
    pl.DataFrame(
        schema={'child': pl.String, 'p_ta': pl.Int64, 'sum_parent_tf': pl.Float64}
    ).write_parquet(empty_path)
    empty_before = empty_path.stat()
    sibling_before = {
        unit.unit_id: (unit_dir / f'{unit.unit_id}.parquet').stat() for unit in nonempty
    }
    (stages / 'parent_contrib.parquet').unlink()
    manifest_path = stages / SCORE_MANIFEST_NAME
    loaded = ScoreStageManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    _ = manifest_path.write_text(
        loaded.model_copy(
            update={
                'completed': tuple(name for name in loaded.completed if name != 'parent_contrib')
            }
        ).model_dump_json()
        + '\n',
        encoding='utf-8',
    )
    score_term_parquet(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    empty_after = empty_path.stat()
    assert empty_after.st_ino == empty_before.st_ino
    assert empty_after.st_mtime_ns == empty_before.st_mtime_ns
    for unit_id, stat in sibling_before.items():
        after = (unit_dir / f'{unit_id}.parquet').stat()
        assert after.st_ino == stat.st_ino
        assert after.st_mtime_ns == stat.st_mtime_ns
    assert (stages / 'parent_contrib.parquet').is_file()


def test_parent_windows_physical_length_join_before_unnest() -> None:
    def explain_physical(expr: ibis.Table) -> str:
        backend = expr._find_backend(use_default=True)
        cursor = backend.raw_sql(f'EXPLAIN {expr.compile()}')
        return '\n'.join(str(row[-1]) for row in cursor.fetchall())

    backend = ibis.duckdb.connect()
    try:
        stats = backend.create_table(
            'stats',
            {
                'term': [
                    'alpha beta gamma delta',
                    'alpha beta gamma',
                    'alpha beta',
                    'coil spring assembly',
                    'coil spring',
                ],
                'tf': [1, 1, 1, 1, 1],
                'df': [1, 1, 1, 1, 1],
                'word_count': [4, 3, 2, 3, 2],
            },
        )
        windows = parent_windows(stats, candidate_span_lengths(stats))
        plan = explain_physical(windows).upper()
    finally:
        backend.disconnect()
    collapsed = re.sub(r'\s+', ' ', plan)
    assert 'UNNEST' in collapsed
    assert 'CROSS_PRODUCT' not in collapsed
    assert 'PIECEWISE_MERGE_JOIN' in collapsed or 'NESTED_LOOP_JOIN' in collapsed
    assert re.search(r'(CHUNK_N\s*>\s*SPAN_N|SPAN_N\s*<\s*CHUNK_N)', collapsed)
    join_at = min(
        index
        for index in (
            collapsed.find('PIECEWISE_MERGE_JOIN'),
            collapsed.find('NESTED_LOOP_JOIN'),
        )
        if index >= 0
    )
    assert collapsed.index('UNNEST') < join_at


def test_planner_keeps_declared_hash_slots_and_peels_only_over_tail() -> None:
    rows = [
        {
            'term': f'slot{slot:02d}-p{index}',
            'parent_span_n': 20,
            'nbytes': 80,
            'estimated_windows': 200_000,
            'estimated_weighted_bytes': 80_000,
            'slot': slot,
        }
        for slot in range(32)
        for index in range(3)
    ]
    rows.append({
        'term': ' '.join(f'w{index}' for index in range(10062)),
        'parent_span_n': 10062,
        'nbytes': 21966,
        'estimated_windows': 2_639_056,
        'estimated_weighted_bytes': 1_250_000_000,
        'slot': 0,
    })
    costs = pl.DataFrame(rows)
    span_values = tuple(range(2, 271))
    ate = _score_ate(parent_buckets=32)
    plan, filters = build_parent_work_plan(costs, span_values, ate=ate)
    hash_units = [unit for unit in plan.units if isinstance(unit, HashSlotUnit)]
    band_units = [unit for unit in plan.units if isinstance(unit, SpanBandUnit)]
    assert len(hash_units) == 32
    assert {unit.slot for unit in hash_units} == set(range(32))
    assert all(unit.total_slots == 32 for unit in hash_units)
    assert all(unit.estimated_windows == 600_000 for unit in hash_units)
    assert band_units
    assert plan.over_cap_filter is not None
    mega = ' '.join(f'w{index}' for index in range(10062))
    assert filters[plan.over_cap_filter] == (mega,)
    assert all(filters[unit.filter_artifact] == (mega,) for unit in band_units)
    assert plan.full_candidate_scans == 32
    assert plan.band_filtered_scans == len(band_units)
    assert plan.full_candidate_scans + plan.band_filtered_scans == len(plan.units)
    assert len(plan.units) <= ate.max_parent_units
    tight = _score_ate(parent_buckets=32, max_parent_units=32)
    with pytest.raises(ParentWorkPlanError, match='max_parent_units'):
        build_parent_work_plan(costs, span_values, ate=tight)
    scan_tight = _score_ate(parent_buckets=32, max_candidate_scans=32)
    with pytest.raises(ParentWorkPlanError, match='max_candidate_scans'):
        build_parent_work_plan(costs, span_values, ate=scan_tight)


def test_planner_fails_closed_when_base_slot_exceeds_aggregate_caps() -> None:
    costs = pl.DataFrame([
        {
            'term': f'heavy{index:02d}',
            'parent_span_n': 40,
            'nbytes': 80,
            'estimated_windows': 400_000,
            'estimated_weighted_bytes': 1_000,
            'slot': 0,
        }
        for index in range(70)
    ])
    ate = _score_ate(parent_buckets=32)
    with pytest.raises(ParentWorkPlanError, match='base aggregate caps'):
        build_parent_work_plan(costs, tuple(range(2, 40)), ate=ate)


def test_plan_term_score_does_not_open_range_or_unit_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError('plan_term_score must not generate windows')

    monkeypatch.setattr('patent_ate.cvalue.algebra.parent_windows', boom)
    monkeypatch.setattr('patent_ate.cvalue.algebra.parent_contributions', boom)
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    planned = plan_term_score(
        extract,
        ate=_score_ate(parent_buckets=4),
        temp_dir=dest / 'spill',
        artifact_dir=dest,
    )
    assert isinstance(planned, ParentScorePlan)
    stages = planned.stages_dir
    unit_dir = stages / PARENT_UNIT_DIRNAME
    assert not list(unit_dir.glob('*.parquet'))
    assert not (stages / 'parent_contrib.parquet').exists()
    assert not (stages / 'scored.parquet').exists()
    assert (stages / PARENT_PLAN_NAME).is_file()
    assert (stages / f'{PARENT_SPAN_COSTS_NAME}.parquet').is_file()
    assert (stages / f'{CANDIDATE_SPAN_LENGTHS_NAME}.parquet').is_file()
    assert (stages / 'term_stats.parquet').is_file()
    plan = planned.work
    assert plan.full_candidate_scans + plan.band_filtered_scans == len(plan.units)
    assert plan.full_candidate_scans <= 4
    if plan.over_cap_filter is not None:
        assert (stages / plan.over_cap_filter).is_file()


def _hybrid_tail_costs() -> tuple[pl.DataFrame, tuple[int, ...], pl.DataFrame]:
    short = tuple(range(1, 10))
    long = (343, 400, 900, 934, 2000, 5000, 7964)
    rows = [
        {
            'term': f'slot{slot:02d}-p0',
            'parent_span_n': 20,
            'nbytes': 80,
            'estimated_windows': 200_000,
            'estimated_weighted_bytes': 80_000,
            'slot': slot,
        }
        for slot in range(32)
    ]
    rows.append({
        'term': ' '.join(f'w{index}' for index in range(10062)),
        'parent_span_n': 10062,
        'nbytes': 21966,
        'estimated_windows': 2_639_056,
        'estimated_weighted_bytes': 1_250_000_000,
        'slot': 1,
    })
    census_rows = [
        {'span_n': length, 'candidate_rows': 8_000_000, 'candidate_bytes': 80_000_000}
        for length in short
    ]
    census_rows.extend([
        {'span_n': 343, 'candidate_rows': 20, 'candidate_bytes': 8_000},
        {'span_n': 400, 'candidate_rows': 10, 'candidate_bytes': 4_000},
        {'span_n': 900, 'candidate_rows': 8, 'candidate_bytes': 3_000},
        {'span_n': 934, 'candidate_rows': 6, 'candidate_bytes': 2_000},
        {'span_n': 2000, 'candidate_rows': 4, 'candidate_bytes': 2_000},
        {'span_n': 5000, 'candidate_rows': 2, 'candidate_bytes': 1_000},
        {'span_n': 7964, 'candidate_rows': 2, 'candidate_bytes': 1_000},
    ])
    return pl.DataFrame(rows), short + long, pl.DataFrame(census_rows)


def _hybrid_tail_ate(**overrides: Any) -> AteSpec:
    values = {
        'parent_buckets': 32,
        'tail_window_cap': 60_000,
        'tail_weighted_bytes_cap': 400_000_000,
    }
    values.update(overrides)
    return _score_ate(**values)


def test_planner_selects_candidate_for_tiny_long_bands_and_window_for_short() -> None:
    costs, span_values, census = _hybrid_tail_costs()
    ate = _hybrid_tail_ate()
    plan, _ = build_parent_work_plan(costs, span_values, ate=ate, census=census)
    bands = [unit for unit in plan.units if isinstance(unit, SpanBandUnit)]
    long_bands = [unit for unit in bands if unit.span_n_min >= 343]
    short_bands = [unit for unit in bands if unit.span_n_max <= 9]
    assert long_bands
    assert short_bands
    assert all(unit.strategy == 'candidate' for unit in long_bands)
    assert all(unit.estimated_candidate_rows <= ate.tail_candidate_row_cap for unit in long_bands)
    assert all(unit.estimated_comparisons < unit.estimated_windows for unit in long_bands)
    assert all(unit.strategy == 'window' for unit in short_bands)
    assert all(unit.estimated_comparisons > unit.estimated_windows for unit in short_bands)


def test_planner_candidate_caps_fail_closed_to_window_or_error() -> None:
    costs, span_values, census = _hybrid_tail_costs()
    windowed, _ = build_parent_work_plan(
        costs,
        span_values,
        ate=_hybrid_tail_ate(tail_candidate_row_cap=1, tail_compare_cap=1),
        census=census,
    )
    assert all(
        unit.strategy == 'window' for unit in windowed.units if isinstance(unit, SpanBandUnit)
    )
    with pytest.raises(ParentWorkPlanError, match='atomic parent length'):
        build_parent_work_plan(
            costs,
            (100,),
            ate=_score_ate(parent_buckets=32, tail_window_cap=10, tail_weighted_bytes_cap=10),
            census=census,
        )


def _edge_term_rows() -> list[dict[str, Any]]:
    return [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': [
                {
                    'term': 'clockwork extra work extra',
                    'frequency': 3,
                    'surfaces': ['clockwork extra work extra'],
                },
                {'term': 'work extra', 'frequency': 2, 'surfaces': ['work extra']},
                {'term': 'clockwork extra', 'frequency': 1, 'surfaces': ['clockwork extra']},
                {
                    'term': 'anti-lock brake system',
                    'frequency': 1,
                    'surfaces': ['anti-lock brake system'],
                },
                {'term': 'lock brake', 'frequency': 2, 'surfaces': ['lock brake']},
                {'term': 'anti lock', 'frequency': 1, 'surfaces': ['anti lock']},
                {'term': 'foo/bar extra', 'frequency': 2, 'surfaces': ['foo/bar extra']},
                {
                    'term': 'pre-foo-bar extra pre-foo/bar extra',
                    'frequency': 1,
                    'surfaces': ['pre-foo-bar extra pre-foo/bar extra'],
                },
                {'term': 'foo+bar extra word', 'frequency': 1, 'surfaces': ['foo+bar extra word']},
                {'term': 'foo+bar extra', 'frequency': 2, 'surfaces': ['foo+bar extra']},
                {'term': 'v1.1 extra word', 'frequency': 1, 'surfaces': ['v1.1 extra word']},
                {'term': 'v1.1 extra', 'frequency': 2, 'surfaces': ['v1.1 extra']},
                {
                    'term': 'coil  spring assembly',
                    'frequency': 1,
                    'surfaces': ['coil  spring assembly'],
                },
                {'term': 'coil  spring', 'frequency': 2, 'surfaces': ['coil  spring']},
                {'term': 'coil spring', 'frequency': 1, 'surfaces': ['coil spring']},
                {'term': 'foo_bar extra word', 'frequency': 1, 'surfaces': ['foo_bar extra word']},
                {'term': 'foo_bar extra', 'frequency': 2, 'surfaces': ['foo_bar extra']},
                {'term': "don't stop extra", 'frequency': 1, 'surfaces': ["don't stop extra"]},
                {'term': "don't stop", 'frequency': 2, 'surfaces': ["don't stop"]},
                {'term': 'iso9001 extra word', 'frequency': 1, 'surfaces': ['iso9001 extra word']},
                {'term': 'iso9001 extra', 'frequency': 2, 'surfaces': ['iso9001 extra']},
                {'term': 'a a a', 'frequency': 4, 'surfaces': ['a a a']},
                {'term': 'a a', 'frequency': 2, 'surfaces': ['a a']},
                {
                    'term': 'the -lock brake sensor extra',
                    'frequency': 1,
                    'surfaces': ['the -lock brake sensor extra'],
                },
                {
                    'term': '-lock brake sensor',
                    'frequency': 2,
                    'surfaces': ['-lock brake sensor'],
                },
                {
                    'term': 'anti-lock brake extra',
                    'frequency': 1,
                    'surfaces': ['anti-lock brake extra'],
                },
                {
                    'term': 'modified (hydroxy)alkyl acrylate monomer extra',
                    'frequency': 1,
                    'surfaces': ['modified (hydroxy)alkyl acrylate monomer extra'],
                },
                {
                    'term': '(hydroxy)alkyl acrylate monomer',
                    'frequency': 2,
                    'surfaces': ['(hydroxy)alkyl acrylate monomer'],
                },
                {
                    'term': 'active nfκb pathway extra',
                    'frequency': 1,
                    'surfaces': ['active nfκb pathway extra'],
                },
                {'term': 'nfκb pathway', 'frequency': 2, 'surfaces': ['nfκb pathway']},
                {
                    'term': 'NFκB pathway extra',
                    'frequency': 1,
                    'surfaces': ['NFκB pathway extra'],
                },
                {'term': 'B pathway', 'frequency': 1, 'surfaces': ['B pathway']},
                {
                    'term': '\u03b1helix structure extra',
                    'frequency': 1,
                    'surfaces': ['\u03b1helix structure extra'],
                },
                {
                    'term': '\u03b1helix structure',
                    'frequency': 2,
                    'surfaces': ['\u03b1helix structure'],
                },
                {'term': 'helix structure', 'frequency': 1, 'surfaces': ['helix structure']},
                {
                    'term': 'pre a.+*(b) extra word',
                    'frequency': 1,
                    'surfaces': ['pre a.+*(b) extra word'],
                },
                {'term': 'a.+*(b) extra', 'frequency': 2, 'surfaces': ['a.+*(b) extra']},
                {
                    'term': 'foo\nline extra\nbar',
                    'frequency': 1,
                    'surfaces': ['foo\nline extra\nbar'],
                },
                {'term': 'line extra', 'frequency': 2, 'surfaces': ['line extra']},
                {
                    'term': 'cafe\u0301 extra word',
                    'frequency': 1,
                    'surfaces': ['cafe\u0301 extra word'],
                },
                {'term': 'extra word', 'frequency': 2, 'surfaces': ['extra word']},
            ],
        }
    ]


def _python_jate_contains(parent: str, child: str) -> bool:
    return bool(re.search(rf'(?<!\w){re.escape(child)}(?!\w)', parent))


def _jate_contrib(rows: list[dict[str, Any]]) -> pl.DataFrame:
    tf: dict[str, int] = {}
    for row in rows:
        for item in row['terms']:
            term = str(item['term'])
            tf[term] = tf.get(term, 0) + int(item['frequency'])
    records = []
    for child in tf:
        child_words = len(child.strip().split())
        parents = [
            parent
            for parent in tf
            if len(parent.strip().split()) > child_words and _python_jate_contains(parent, child)
        ]
        if not parents:
            continue
        records.append({
            'child': child,
            'p_ta': len(parents),
            'sum_parent_tf': float(sum(tf[parent] for parent in parents)),
        })
    if not records:
        return pl.DataFrame(
            schema={'child': pl.String, 'p_ta': pl.Int64, 'sum_parent_tf': pl.Float64}
        )
    return pl.DataFrame(records).with_columns(
        pl.col('p_ta').cast(pl.Int64),
        pl.col('sum_parent_tf').cast(pl.Float64),
    )


def _jate_containment_build_contrib(rows: list[dict[str, Any]]) -> pl.DataFrame:
    pool: dict[str, Candidate] = {}
    tf: dict[str, int] = {}
    for row in rows:
        for item in row['terms']:
            term = str(item['term'])
            cand = pool.setdefault(term, Candidate(surface_form=term, normalized_form=term))
            cand.surface_forms.update(item['surfaces'])
            tf[term.lower()] = tf.get(term.lower(), 0) + int(item['frequency'])
            for index in range(int(item['frequency'])):
                cand.add_position(str(row[PATENT_ID_COLUMN]), index, index + 1, 0)
    candidates = list(pool.values())
    containment = Containment.build(candidates, TermComponentIndex.build(candidates))
    records = [
        {
            'child': child,
            'p_ta': len(parents),
            'sum_parent_tf': float(sum(tf[parent] for parent in parents)),
        }
        for child, parents in containment.term2parents.items()
        if parents
    ]
    if not records:
        return pl.DataFrame(
            schema={'child': pl.String, 'p_ta': pl.Int64, 'sum_parent_tf': pl.Float64}
        )
    return pl.DataFrame(records).with_columns(
        pl.col('p_ta').cast(pl.Int64),
        pl.col('sum_parent_tf').cast(pl.Float64),
    )


def test_candidate_contributions_match_window_and_jate_on_edge_corpus() -> None:
    rows = _edge_term_rows()
    stats = term_stats(_patents(rows))
    windowed = parent_contributions(stats).to_polars()
    parents = stats.filter(stats.word_count >= 2)
    lengths = candidate_span_lengths(stats).to_polars()
    span_min = int(lengths['span_n'].min())
    span_max = int(lengths['span_n'].max())
    candidate = candidate_parent_contributions(
        stats,
        parents=parents,
        span_n_min=span_min,
        span_n_max=span_max,
    ).to_polars()
    assert _sorted_frame(candidate, 'child').equals(_sorted_frame(windowed, 'child'))
    later = candidate.filter(pl.col('child') == 'work extra')
    assert later.height == 1
    assert int(later['p_ta'][0]) == 1
    assert float(later['sum_parent_tf'][0]) == pytest.approx(3.0)
    repeated = candidate.filter(pl.col('child') == 'a a')
    assert repeated.height == 1
    assert int(repeated['p_ta'][0]) == 1
    expected = _jate_contrib(rows)
    assert _sorted_frame(candidate, 'child').equals(_sorted_frame(expected, 'child'))
    lowered_rows = [
        {
            **row,
            'terms': [
                {
                    **item,
                    'term': str(item['term']).lower(),
                    'surfaces': [str(surface).lower() for surface in item['surfaces']],
                }
                for item in row['terms']
            ],
        }
        for row in rows
    ]
    lowered_expected = _jate_containment_build_contrib(lowered_rows)
    assert _sorted_frame(_jate_contrib(lowered_rows), 'child').equals(
        _sorted_frame(lowered_expected, 'child')
    )
    hits = {str(row['child']): int(row['p_ta']) for row in candidate.iter_rows(named=True)}
    assert hits['-lock brake sensor'] == 1
    assert hits['(hydroxy)alkyl acrylate monomer'] == 1
    assert hits['nfκb pathway'] == 1
    assert hits['\u03b1helix structure'] == 1
    assert 'helix structure' not in hits
    assert 'B pathway' not in hits
    assert hits['a.+*(b) extra'] == 1
    assert hits['line extra'] == 1


def test_candidate_sql_uses_contains_escape_and_nested_loop() -> None:
    def explain_physical(expr: ibis.Table) -> str:
        backend = expr._find_backend(use_default=True)
        cursor = backend.raw_sql(f'EXPLAIN {expr.compile()}')
        return '\n'.join(str(row[-1]) for row in cursor.fetchall())

    backend = ibis.duckdb.connect()
    try:
        stats = backend.create_table(
            'edge_stats',
            {
                'term': [
                    'clockwork extra work extra',
                    'work extra',
                    'anti-lock brake system',
                    'lock brake',
                ],
                'tf': [3, 2, 1, 2],
                'df': [1, 1, 1, 1],
                'word_count': [4, 2, 3, 2],
            },
        )
        expr = candidate_parent_contributions(
            stats,
            parents=stats.filter(stats.word_count >= 2),
            span_n_min=1,
            span_n_max=4,
        )
        raw_sql = ibis.to_sql(expr)
        sql = raw_sql.upper()
        plan = explain_physical(expr).upper()
    finally:
        backend.disconnect()
    assert 'CONTAINS' in sql
    assert 'REGEXP_ESCAPE' in sql
    assert 'REGEXP_MATCHES' in sql
    assert JATE_WORD_CLASS in raw_sql
    assert r'\w+|\W+' not in raw_sql
    assert r'\W' not in raw_sql.replace(JATE_WORD_CLASS, '').replace(JATE_CHUNK_PATTERN, '')
    assert 'RANGE' not in sql
    assert 'NESTED_LOOP_JOIN' in plan
    assert 'RANGE' not in plan


@pytest.mark.parametrize('stale_semantic', ['4', '5'])
def test_resume_rejects_semantic4_and_5_units(tmp_path: Path, stale_semantic: str) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    ate = _score_ate(parent_buckets=4)
    first = plan_term_score(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    stages = first.stages_dir
    stats_path = stages / 'term_stats.parquet'
    surfaces_path = stages / 'surfaces.parquet'
    assert not surfaces_path.is_file()
    dummy_surfaces = pl.DataFrame({'term': ['coil spring'], 'key': ['coil spring']})
    dummy_surfaces.write_parquet(surfaces_path)
    stats_before = stats_path.stat()
    surfaces_before = surfaces_path.stat()
    unit_dir = stages / PARENT_UNIT_DIRNAME
    dummy = pl.DataFrame(
        {'child': ['coil spring'], 'p_ta': [1], 'sum_parent_tf': [1.0]},
        schema={'child': pl.String, 'p_ta': pl.Int64, 'sum_parent_tf': pl.Float64},
    )
    stale_ids = first.work.unit_ids() or ('hash-0000',)
    unit_dir.mkdir(parents=True, exist_ok=True)

    def persist_stale(unit_id: str) -> None:
        dummy.write_parquet(unit_dir / f'{unit_id}.parquet')

    tuple(map(persist_stale, stale_ids))
    stale_bytes = {unit_id: (unit_dir / f'{unit_id}.parquet').read_bytes() for unit_id in stale_ids}
    lengths_path = stages / f'{CANDIDATE_SPAN_LENGTHS_NAME}.parquet'
    costs_path = stages / f'{PARENT_SPAN_COSTS_NAME}.parquet'
    _poison_stage(lengths_path)
    _poison_stage(costs_path)
    manifest = ScoreStageManifest.model_validate_json(
        (stages / SCORE_MANIFEST_NAME).read_text(encoding='utf-8')
    )
    _ = (stages / SCORE_MANIFEST_NAME).write_text(
        manifest.model_copy(
            update={
                'semantic_version': stale_semantic,
                'completed': (
                    'term_stats',
                    'surfaces',
                    CANDIDATE_SPAN_LENGTHS_NAME,
                    PARENT_SPAN_COSTS_NAME,
                ),
                'completed_units': stale_ids,
            }
        ).model_dump_json()
        + '\n',
        encoding='utf-8',
    )
    restored = plan_term_score(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    restored_manifest = ScoreStageManifest.model_validate_json(
        (stages / SCORE_MANIFEST_NAME).read_text(encoding='utf-8')
    )
    assert restored_manifest.semantic_version == SCORE_SEMANTIC_VERSION
    assert restored.completed_units == ()
    assert stats_path.stat().st_ino == stats_before.st_ino
    assert surfaces_path.stat().st_ino == surfaces_before.st_ino
    _assert_parquet_replaced(lengths_path)
    _assert_parquet_replaced(costs_path)
    assert all(
        not (unit_dir / f'{unit_id}.parquet').is_file()
        or (unit_dir / f'{unit_id}.parquet').read_bytes() != payload
        for unit_id, payload in stale_bytes.items()
    )


def test_strategy_cap_change_invalidates_unit_resume(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest = tmp_path / 'run'
    _write_compact(extract / 'part-0.parquet', _coil_rows())
    ate = _score_ate(parent_buckets=4)
    first = plan_term_score(extract, ate=ate, temp_dir=dest / 'spill', artifact_dir=dest)
    stages = first.stages_dir
    unit_dir = stages / PARENT_UNIT_DIRNAME
    dummy = pl.DataFrame(
        {'child': ['coil spring'], 'p_ta': [1], 'sum_parent_tf': [1.0]},
        schema={'child': pl.String, 'p_ta': pl.Int64, 'sum_parent_tf': pl.Float64},
    )

    def persist_unit(unit: HashSlotUnit | SpanBandUnit) -> None:
        dummy.write_parquet(unit_dir / f'{unit.unit_id}.parquet')

    tuple(map(persist_unit, first.work.units))
    stale_bytes = {
        unit.unit_id: (unit_dir / f'{unit.unit_id}.parquet').read_bytes()
        for unit in first.work.units
    }
    manifest = ScoreStageManifest.model_validate_json(
        (stages / SCORE_MANIFEST_NAME).read_text(encoding='utf-8')
    )
    _ = (stages / SCORE_MANIFEST_NAME).write_text(
        manifest.model_copy(update={'completed_units': first.work.unit_ids()}).model_dump_json()
        + '\n',
        encoding='utf-8',
    )
    second = plan_term_score(
        extract,
        ate=_score_ate(parent_buckets=4, tail_candidate_row_cap=1),
        temp_dir=dest / 'spill',
        artifact_dir=dest,
    )
    assert second.completed_units == ()
    assert all(
        not (unit_dir / f'{unit_id}.parquet').is_file()
        or (unit_dir / f'{unit_id}.parquet').read_bytes() != payload
        for unit_id, payload in stale_bytes.items()
    )


def test_hybrid_compact_merge_unchanged(tmp_path: Path) -> None:
    extract = tmp_path / 'extract'
    dest_window = tmp_path / 'window'
    dest_hybrid = tmp_path / 'hybrid'
    _write_compact(extract / 'part-0.parquet', _edge_term_rows())
    windowed = TermhoodStore.open(
        score_term_parquet(
            extract,
            ate=_score_ate(tail_candidate_row_cap=1, tail_compare_cap=1),
            temp_dir=dest_window / 'spill',
            artifact_dir=dest_window,
        )
    )
    hybrid = TermhoodStore.open(
        score_term_parquet(
            extract,
            ate=_score_ate(),
            temp_dir=dest_hybrid / 'spill',
            artifact_dir=dest_hybrid,
        )
    )
    left = pl.scan_parquet(windowed.parquet).collect().sort('key')
    right = pl.scan_parquet(hybrid.parquet).collect().sort('key')
    assert left.equals(right)
    hybrid_plan = ParentWorkPlan.model_validate_json(
        (dest_hybrid / SCORE_STAGE_DIRNAME / PARENT_PLAN_NAME).read_text(encoding='utf-8')
    )
    contrib = pl.scan_parquet(
        dest_hybrid / SCORE_STAGE_DIRNAME / 'parent_contrib.parquet'
    ).collect()
    assert set(contrib.columns) == {'child', 'p_ta', 'sum_parent_tf'}
    assert hybrid_plan.units
    census = candidate_span_census(term_stats(_patents(_edge_term_rows()))).to_polars()
    assert census.height >= 1


def test_window_sql_length_before_range_and_boundary_after_join() -> None:
    stats = term_stats(_patents(_coil_rows()))
    expr = parent_contributions(stats)
    sql = ibis.to_sql(expr)
    upper = sql.upper()
    assert 'SPAN_N' in upper
    assert 'RANGE' in upper
    assert upper.index('SPAN_N') < upper.index('RANGE')
    assert 'WINDOW' in upper
    assert 'REGEXP_ESCAPE' in upper
    assert 'REGEXP_MATCHES' in upper
    assert upper.index('WINDOW') < upper.index('REGEXP_MATCHES')
    assert JATE_WORD_CLASS in sql
    assert r'\w+|\W+' not in sql


def test_shared_containment_matches_python_jate_word_class() -> None:
    pairs = [
        ('the -lock brake sensor extra', '-lock brake sensor', True),
        ('anti-lock brake extra', '-lock brake', False),
        ('modified (hydroxy)alkyl acrylate monomer extra', '(hydroxy)alkyl acrylate monomer', True),
        ('active nfκb pathway extra', 'nfκb pathway', True),
        ('NFκB pathway extra', 'B pathway', False),
        ('\u03b1helix structure extra', '\u03b1helix structure', True),
        ('\u03b1helix structure extra', 'helix structure', False),
        ('pre a.+*(b) extra word', 'a.+*(b) extra', True),
        ('foo\nline extra\nbar', 'line extra', True),
        ('clockwork extra work extra', 'work extra', True),
        ('cafe\u0301 extra word', 'extra word', True),
    ]
    backend = ibis.duckdb.connect()
    try:
        table = backend.create_table(
            'pairs',
            {
                'parent': [parent for parent, _, _ in pairs],
                'child': [child for _, child, _ in pairs],
            },
        )
        frame = table.mutate(hit=jate_contains(table.parent, table.child)).to_polars()
    finally:
        backend.disconnect()
    for index, (parent, child, expected) in enumerate(pairs):
        python_hit = _python_jate_contains(parent, child)
        assert python_hit is expected
        assert bool(frame['hit'][index]) is expected


_EXACTNESS_PHRASES = (
    '-lock brake sensor',
    '(hydroxy)alkyl acrylate monomer',
    'nfκb pathway',
    '\u03b1helix structure',
    'helix structure',
    'B pathway',
    'coil spring',
    'coil spring assembly',
    'a.+*(b) extra',
)


@given(
    phrases=st.lists(st.sampled_from(_EXACTNESS_PHRASES), min_size=2, max_size=6, unique=True),
)
@settings(max_examples=15, deadline=None)
def test_window_candidate_and_jate_agree_on_generated_phrases(phrases: list[str]) -> None:
    extra = [f'context {phrase} trailing' for phrase in phrases]
    rows = [
        {
            PATENT_ID_COLUMN: 'p1',
            'n_docs': 1,
            'terms': [
                {'term': term, 'frequency': 1 + index, 'surfaces': [term]}
                for index, term in enumerate([*extra, *phrases])
            ],
        }
    ]
    stats = term_stats(_patents(rows))
    windowed = parent_contributions(stats).to_polars()
    lengths = candidate_span_lengths(stats).to_polars()
    candidate = candidate_parent_contributions(
        stats,
        parents=stats.filter(stats.word_count >= 2),
        span_n_min=int(lengths['span_n'].min()),
        span_n_max=int(lengths['span_n'].max()),
    ).to_polars()
    assert _sorted_frame(candidate, 'child').equals(_sorted_frame(windowed, 'child'))
    assert _sorted_frame(candidate, 'child').equals(_sorted_frame(_jate_contrib(rows), 'child'))

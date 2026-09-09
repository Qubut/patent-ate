"""Plan parent work units and score C-value from compact extract Parquet.

Bind the dest, plan units, then score and commit. A restarted job reuses
Parquet that is still valid under the current settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import compress
from operator import attrgetter, methodcaller
from pathlib import Path

import polars as pl
import structlog

from patent_ate.spec import AteSpec
from patent_ate.termhood import TermhoodStore

from . import algebra
from . import index as indexed
from .exec import ScoreStageExecutor
from .plan import (
    PARENT_PLAN_NAME,
    PARENT_SPAN_COSTS_NAME,
    IndexedContainmentPlan,
    ParentWorkPlan,
    build_parent_work_plan,
)
from .store import (
    CANDIDATE_SPAN_LENGTHS_NAME,
    COSTS_CAST,
    SCORE_SEMANTIC_VERSION,
    SPAN_LENGTHS_CAST,
    SURFACES_CAST,
    ExtractFingerprint,
    ScoreStageSession,
    ScoreStageStore,
)

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ParentScorePlan:
    """Materialized lengths, parent costs, and a bounded work plan."""

    work: ParentWorkPlan | None
    dest: Path
    stages_dir: Path
    extract_files: tuple[Path, ...]
    fingerprint: ExtractFingerprint
    total_docs: int
    resume_units: bool
    completed_units: tuple[str, ...]
    indexed: IndexedContainmentPlan | None = None


def load_or_build_window_plan(
    session: ScoreStageSession,
    store: ScoreStageStore,
    executor: ScoreStageExecutor,
    ate: AteSpec,
) -> ParentWorkPlan:
    """Reuse a matching window plan or peel a new one from parent costs."""
    loaded = store.load_matching_plan(ate) if session.resume_units else None
    if loaded is not None:
        return loaded
    if not session.resume_units:
        store.wipe_units_and_scored()
        session.finished_units.clear()
    census = executor.span_census(session)
    costs = pl.read_parquet(store.stage_parquet(PARENT_SPAN_COSTS_NAME))
    span_values = tuple(
        sorted(
            pl
            .scan_parquet(store.stage_parquet(CANDIDATE_SPAN_LENGTHS_NAME))
            .select(pl.col('span_n').cast(pl.Int64))
            .collect()['span_n']
            .to_list()
        )
    )
    built, filter_terms = build_parent_work_plan(costs, span_values, ate=ate, census=census)
    store.write_filters(filter_terms, resume_units=session.resume_units)
    store.replace_plan(built)
    if session.resume_units:
        ids = tuple(map(attrgetter('unit_id'), built.units))
        kept = frozenset(filter(store.unit_valid, filter(session.finished_units.__contains__, ids)))
        session.finished_units[:] = list(filter(kept.__contains__, ids))
        session.finish(PARENT_SPAN_COSTS_NAME, tuple(session.finished_units))
    else:
        session.finish(PARENT_SPAN_COSTS_NAME, ())
    return built


def plan_term_score(
    extract_dir: Path,
    *,
    ate: AteSpec,
    temp_dir: Path,
    artifact_dir: Path | None = None,
) -> ParentScorePlan:
    """Write or reuse term stats, candidate lengths, and parent costs, then plan units.

    Window strings and unit contribution files are written later during score.
    """
    dest = artifact_dir if artifact_dir is not None else extract_dir
    dest.mkdir(parents=True, exist_ok=True)
    store = ScoreStageStore(dest)
    files = store.extract_files(extract_dir)
    if not files:
        raise ValueError('compact extract parquet is required to plan C-value scoring')
    TermhoodStore.clear_partials(dest)
    executor = ScoreStageExecutor(ate, temp_dir)
    executor.clear_spill()
    session = store.bind(extract_dir, ate)
    executor.materialize_term_stats(session)
    if ate.containment == 'indexed':
        indexed_plan, resume_indexed = indexed.resume_or_write_plan(
            session, ate, semantic_version=SCORE_SEMANTIC_VERSION
        )
        return ParentScorePlan(
            work=None,
            dest=dest,
            stages_dir=store.stages_dir,
            extract_files=files,
            fingerprint=session.fingerprint,
            total_docs=session.total_docs,
            resume_units=resume_indexed,
            completed_units=tuple(session.finished_units),
            indexed=indexed_plan,
        )
    executor.materialize_stage(
        session,
        CANDIDATE_SPAN_LENGTHS_NAME,
        lambda connection: algebra.candidate_span_lengths(
            connection.read_parquet(session.store.stage_parquet('term_stats'))
        ),
        SPAN_LENGTHS_CAST,
    )
    reuse_costs = (
        session.resume_units
        and session.stage_valid(PARENT_SPAN_COSTS_NAME)
        and session.prior_parent_buckets == ate.parent_buckets
    )
    if not reuse_costs:
        session.store.stage_parquet(PARENT_SPAN_COSTS_NAME).unlink(missing_ok=True)
    executor.materialize_stage(
        session,
        PARENT_SPAN_COSTS_NAME,
        lambda connection: algebra.parent_span_costs(
            connection.read_parquet(session.store.stage_parquet('term_stats')),
            connection.read_parquet(session.store.stage_parquet(CANDIDATE_SPAN_LENGTHS_NAME)),
            buckets=ate.parent_buckets,
        ),
        COSTS_CAST,
        resume=reuse_costs,
    )

    plan = load_or_build_window_plan(session, store, executor, ate)
    dumped = tuple(map(methodcaller('model_dump'), plan.units))
    span_dumps = tuple(
        compress(dumped, map('span_band'.__eq__, map(methodcaller('get', 'kind'), dumped)))
    )
    bands = pl.DataFrame(list(span_dumps))
    peeled_path = None if plan.over_cap_filter is None else store.stages_dir / plan.over_cap_filter
    peeled = (
        0
        if peeled_path is None
        else int(pl.scan_parquet(peeled_path).select(pl.len()).collect().item())
    )
    n_candidate = 0 if not span_dumps else bands.filter(pl.col('strategy') == 'candidate').height
    n_window = 0 if not span_dumps else bands.filter(pl.col('strategy') == 'window').height
    max_cmp = (
        0
        if not span_dumps
        else int(bands.select(pl.col('estimated_comparisons').max().fill_null(0)).item())
    )
    _log.info(
        'patent_ate.score.plan',
        n_units=len(plan.units),
        n_hash=plan.full_candidate_scans,
        n_span=plan.band_filtered_scans,
        n_span_candidate=n_candidate,
        n_span_window=n_window,
        n_peeled=peeled,
        max_unit_windows=max(map(attrgetter('estimated_windows'), plan.units), default=0),
        max_unit_weighted_bytes=max(
            map(attrgetter('estimated_weighted_bytes'), plan.units), default=0
        ),
        max_unit_comparisons=max_cmp,
        full_candidate_scans=plan.full_candidate_scans,
        band_filtered_scans=plan.band_filtered_scans,
        max_parent_units=ate.max_parent_units,
        max_candidate_scans=ate.max_candidate_scans,
        tail_candidate_row_cap=ate.tail_candidate_row_cap,
        tail_compare_cap=ate.tail_compare_cap,
    )
    plan_path = store.stages_dir / PARENT_PLAN_NAME
    return ParentScorePlan(
        work=plan,
        dest=dest,
        stages_dir=store.stages_dir,
        extract_files=files,
        fingerprint=session.fingerprint,
        total_docs=session.total_docs,
        resume_units=session.resume_units and plan_path.is_file() and plan.matches(ate),
        completed_units=tuple(session.finished_units),
    )


def score_term_parquet(
    extract_dir: Path,
    *,
    ate: AteSpec,
    temp_dir: Path,
    artifact_dir: Path | None = None,
) -> Path:
    """Score compact extract Parquet into a committed termhood fact table."""
    dest = artifact_dir if artifact_dir is not None else extract_dir
    files = ScoreStageStore.extract_files(extract_dir)
    if not files:
        dest.mkdir(parents=True, exist_ok=True)
        return Path(
            TermhoodStore.write_frame(
                pl.DataFrame(schema={'key': pl.String, 'c_value': pl.Float64, 'df': pl.Int64}),
                dest,
                total_docs=0,
            ).root
        )
    planned = plan_term_score(
        extract_dir,
        ate=ate,
        temp_dir=temp_dir,
        artifact_dir=dest,
    )
    store = ScoreStageStore(planned.dest)
    session = store.attach(
        ate,
        extract_files=planned.extract_files,
        fingerprint=planned.fingerprint,
        resume_units=planned.resume_units,
        completed_units=planned.completed_units,
        total_docs=planned.total_docs,
    )
    executor = ScoreStageExecutor(ate, temp_dir)
    executor.materialize_stage(
        session,
        'surfaces',
        lambda connection: algebra.surfaces(connection.read_parquet(planned.extract_files)),
        SURFACES_CAST,
        resume=session.stage_valid('surfaces'),
    )
    work = planned.work
    missing_contrib = 'parent_contrib' not in session.completed or not session.stage_valid(
        'parent_contrib'
    )
    if missing_contrib and ate.containment == 'indexed':
        executor.write_indexed_contributions(session)
    elif missing_contrib:
        if work is None:
            raise TypeError('window scoring requires a parent work plan')
        executor.write_units(session, work)
        executor.materialize_stage(
            session,
            'parent_contrib',
            lambda connection: executor.merge_contributions(connection, session, work),
            resume=False,
        )
        session.finish('parent_contrib', tuple(map(attrgetter('unit_id'), work.units)))
    elif work is None:
        session.finish('parent_contrib')
    else:
        session.finish('parent_contrib', tuple(map(attrgetter('unit_id'), work.units)))
    executor.write_scored(session)
    return Path(executor.commit_keys(session, dest))

"""DuckDB sessions that materialize Ibis score stages to Parquet.

One backend per spill directory. Finished units are skipped on restart.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from operator import attrgetter, itemgetter, methodcaller
from pathlib import Path

import ibis
import polars as pl
import structlog
from ibis.backends.duckdb import Backend
from ibis.expr.types import Table
from returns.io import IOResultE, impure_safe
from returns.pipeline import is_successful, managed
from returns.result import ResultE
from returns.unsafe import unsafe_perform_io

from patent_ate.spec import AteSpec
from patent_ate.termhood import TERMHOOD_PARQUET_NAME, TermhoodStore

from . import algebra
from . import index as indexed
from .plan import (
    INDEXED_CONTRIB_BATCH_PACKS,
    INDEXED_CONTRIB_BATCH_WORKERS,
    INDEXED_CONTRIB_CHUNKS,
    INDEXED_MAX_CHUNK_COST,
    INDEXED_MAX_INTERVALS_PER_CHUNK,
    PARENT_SPAN_COSTS_NAME,
    SCORE_TERM_BUCKETS,
    SCORED_CHUNK_DIRNAME,
    HashSlotUnit,
    ParentWorkPlan,
    ParentWorkUnit,
    SpanBandUnit,
)
from .store import (
    CANDIDATE_SPAN_LENGTHS_NAME,
    CONTRIB_CAST,
    KEYS_CAST,
    SCORED_CAST,
    STAGE_CASTS,
    STAGE_SCHEMAS,
    TERM_STATS_CAST,
    IndexedMapsFingerprint,
    ScoreStageSession,
)

_log = structlog.get_logger(__name__)


class ScoreStageExecutor:
    """One managed backend plus typed Parquet materialize of a score dest."""

    def __init__(self, ate: AteSpec, temp_dir: Path) -> None:
        self._ate = ate
        self._temp_dir = temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)

    def spill_bytes(self) -> int:
        """Return current DuckDB spill file bytes under the temp directory."""
        files = filter(Path.is_file, self._temp_dir.rglob('*'))
        return sum(map(attrgetter('st_size'), map(Path.stat, files)))

    def clear_spill(self) -> None:
        """Unlink leftover spill files before a new managed session."""
        tuple(
            map(
                methodcaller('unlink', missing_ok=True),
                filter(Path.is_file, self._temp_dir.rglob('*')),
            )
        )

    def run[T](self, use: Callable[[Backend], T]) -> T:
        """Acquire DuckDB, run ``use``, disconnect, and wipe spill."""

        @impure_safe
        def open_backend() -> Backend:
            return ibis.duckdb.connect(
                memory_limit=self._ate.duckdb_memory,
                temp_directory=str(self._temp_dir),
                threads=self._ate.duckdb_threads,
                preserve_insertion_order=False,
            )

        @impure_safe
        def disconnect(connection: Backend, _result: ResultE[T]) -> None:
            connection.disconnect()
            self.clear_spill()

        @impure_safe
        def apply(connection: Backend) -> T:
            return use(connection)

        result: IOResultE[T] = managed(apply, disconnect)(open_backend())
        if not is_successful(result):
            raise unsafe_perform_io(result.failure())
        return unsafe_perform_io(result.unwrap())

    def replace_temp_table(self, connection: Backend, name: str, expr: Table) -> Table:
        """Create a temp table, dropping any previous table of the same name."""
        connection.raw_sql(f'DROP TABLE IF EXISTS "{name}"')
        return connection.create_table(name, expr, temp=True)

    def write_table(
        self,
        connection: Backend,
        expr: Table,
        casts: dict[str, str],
        dest: Path,
    ) -> int:
        """Write ``expr`` through a typed temp table to ``dest``."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = f'_{dest.name.replace(".", "_")}_typed'
        self.replace_temp_table(connection, tmp, expr.cast(casts))
        connection.table(tmp).to_parquet(dest)
        return int(connection.read_parquet(dest).count().to_pyarrow().as_py() or 0)

    def publish_parquet(
        self,
        connection: Backend,
        expr: Table,
        casts: dict[str, str],
        dest: Path,
    ) -> tuple[int, int, int]:
        """Write typed Parquet through ``.partial`` and return rows, bytes, spill."""
        partial = dest.with_name(f'{dest.name}.partial')
        partial.unlink(missing_ok=True)
        n_rows = self.write_table(connection, expr, casts, partial)
        output_bytes = partial.stat().st_size
        spill = self.spill_bytes()
        partial.replace(dest)
        return n_rows, output_bytes, spill

    def materialize_stage(
        self,
        session: ScoreStageSession,
        name: str,
        build: Callable[[Backend], Table],
        casts: dict[str, str] | None = None,
        *,
        resume: bool | None = None,
    ) -> None:
        """Reuse or rewrite one declared stage Parquet and publish progress."""
        dest = session.store.stage_parquet(name)
        should_resume = session.stage_valid(name) if resume is None else resume
        if should_resume:
            session.note_resume(name)
            session.finish(name)
            return
        schema = casts if casts is not None else STAGE_CASTS[name]

        def write(connection: Backend) -> int:
            started = time.perf_counter()
            n_rows, output_bytes, spill = self.publish_parquet(
                connection,
                build(connection),
                schema,
                dest,
            )
            _log.info(
                'patent_ate.score.stage',
                stage=name,
                resumed=False,
                wall_s=round(time.perf_counter() - started, 6),
                n_rows=n_rows,
                output_bytes=output_bytes,
                spill_bytes=spill,
            )
            return n_rows

        _ = self.run(write)
        session.finish(name)

    def materialize_term_stats(self, session: ScoreStageSession) -> None:
        """Write or reuse compact term stats and record ``total_docs``."""
        if session.stage_valid('term_stats'):
            session.note_resume('term_stats')
        else:

            def write(connection: Backend) -> int:
                started = time.perf_counter()
                patents = connection.read_parquet(session.extract_files)
                n_docs = int(patents.n_docs.sum().to_pyarrow().as_py() or 0)
                dest = session.store.stage_parquet('term_stats')
                n_rows, output_bytes, spill = self.publish_parquet(
                    connection,
                    algebra.term_stats(patents),
                    TERM_STATS_CAST,
                    dest,
                )
                _log.info(
                    'patent_ate.score.stage',
                    stage='term_stats',
                    resumed=False,
                    wall_s=round(time.perf_counter() - started, 6),
                    n_rows=n_rows,
                    output_bytes=output_bytes,
                    spill_bytes=spill,
                )
                return n_docs

            session.total_docs = self.run(write)
        if 'term_stats' not in session.completed:
            session.completed.insert(0, 'term_stats')
        session.finish(
            'term_stats',
            tuple(session.finished_units) if session.finished_units else None,
        )

    def span_census(self, session: ScoreStageSession) -> pl.DataFrame:
        """Materialize per-length candidate row and byte counts."""

        def read(connection: Backend) -> pl.DataFrame:
            materialized = algebra.candidate_span_census(
                connection.read_parquet(session.store.stage_parquet('term_stats'))
            ).to_polars()
            if not isinstance(materialized, pl.DataFrame):
                msg = 'candidate span census must materialize as a DataFrame'
                raise TypeError(msg)
            return materialized

        return self.run(read)

    def contributions_for(
        self,
        connection: Backend,
        unit: ParentWorkUnit,
        plan: ParentWorkPlan,
        session: ScoreStageSession,
    ) -> Table:
        """Return compact parent contributions of one declared work unit."""
        stats = connection.read_parquet(session.store.stage_parquet('term_stats'))
        lengths = connection.read_parquet(session.store.stage_parquet(CANDIDATE_SPAN_LENGTHS_NAME))
        exclude = (
            connection.read_parquet(session.store.stages_dir / plan.over_cap_filter)
            if plan.over_cap_filter is not None
            else None
        )
        match unit:
            case HashSlotUnit(total_slots=slots, slot=slot):
                parents = stats.filter(
                    (stats.word_count >= 2) & (algebra.parent_bucket(stats.term, slots) == slot)
                )
                if exclude is not None:
                    parents = parents.anti_join(exclude, 'term')
                return algebra.parent_contributions(stats, lengths, parents=parents)
            case SpanBandUnit(
                strategy='candidate',
                filter_artifact=rel,
                span_n_min=lo,
                span_n_max=hi,
            ):
                parents = stats.inner_join(
                    connection.read_parquet(session.store.stages_dir / rel),
                    'term',
                )
                return algebra.candidate_parent_contributions(
                    stats,
                    parents=parents,
                    span_n_min=lo,
                    span_n_max=hi,
                )
            case SpanBandUnit(
                strategy='window',
                filter_artifact=rel,
                span_n_min=lo,
                span_n_max=hi,
            ):
                parents = stats.inner_join(
                    connection.read_parquet(session.store.stages_dir / rel),
                    'term',
                )
                band_lengths = lengths.filter((lengths.span_n >= lo) & (lengths.span_n <= hi))
                aligned = algebra.chunk_aligned(stats)
                candidates = aligned.filter(
                    (aligned.chunk_n >= lo) & (aligned.chunk_n <= hi)
                ).select('term', 'tf', 'df', 'word_count')
                return algebra.parent_contributions(candidates, band_lengths, parents=parents)
            case _:
                msg = f'unsupported parent work unit: {type(unit)!r}'
                raise TypeError(msg)

    def merge_contribution_parts(self, connection: Backend, parts: tuple[Path, ...]) -> Table:
        """Sum compact contribution Parquet parts by child term."""
        if not parts:
            return ibis.memtable(
                {'child': [], 'p_ta': [], 'sum_parent_tf': []},
                schema={'child': 'string', 'p_ta': 'int64', 'sum_parent_tf': 'float64'},
            )
        return (
            connection
            .read_parquet(parts)
            .group_by('child')
            .agg(
                p_ta=ibis._.p_ta.cast('float64').sum().cast('int64'),
                sum_parent_tf=ibis._.sum_parent_tf.sum().cast('float64'),
            )
        )

    def merge_contributions(
        self,
        connection: Backend,
        session: ScoreStageSession,
        plan: ParentWorkPlan,
    ) -> Table:
        """Sum compact unit contributions by child term."""
        return self.merge_contribution_parts(
            connection,
            tuple(map(session.store.unit_path, map(attrgetter('unit_id'), plan.units))),
        )

    def write_unit(
        self,
        session: ScoreStageSession,
        unit: ParentWorkUnit,
        plan: ParentWorkPlan,
    ) -> None:
        """Materialize one unit contribution file and record its id."""

        def write(connection: Backend) -> int:
            started = time.perf_counter()
            dest = session.store.unit_path(unit.unit_id)
            n_rows, output_bytes, spill = self.publish_parquet(
                connection,
                self.contributions_for(connection, unit, plan, session),
                CONTRIB_CAST,
                dest,
            )
            _log.info(
                'patent_ate.score.stage',
                stage=f'parent_unit_{unit.unit_id}',
                resumed=False,
                strategy=unit.strategy if isinstance(unit, SpanBandUnit) else 'window',
                estimated_windows=unit.estimated_windows,
                estimated_weighted_bytes=unit.estimated_weighted_bytes,
                estimated_candidate_rows=(
                    unit.estimated_candidate_rows if isinstance(unit, SpanBandUnit) else 0
                ),
                estimated_comparisons=(
                    unit.estimated_comparisons if isinstance(unit, SpanBandUnit) else 0
                ),
                wall_s=round(time.perf_counter() - started, 6),
                n_rows=n_rows,
                output_bytes=output_bytes,
                spill_bytes=spill,
            )
            return n_rows

        _ = self.run(write)
        session.finished_units.append(unit.unit_id)
        session.finish(PARENT_SPAN_COSTS_NAME, tuple(session.finished_units))

    def write_indexed_chunk_batch(  # ruff: ignore[complex-structure]
        self,
        session: ScoreStageSession,
        work: Path,
        chunk_dir: Path,
        chunk_ids: Sequence[int],
        n_chunks: int,
        *,
        batch_i: int,
        n_batches: int,
        worker_i: int = 0,
        n_workers: int = 1,
    ) -> None:
        """Write unfinished packs sharing one DuckDB session and one sa_color band."""
        ids = tuple(chunk_ids)
        if not ids:
            return

        def chunk_path(chunk_i: int) -> Path:
            return chunk_dir / f'{chunk_i:05d}.parquet'

        census = (
            pl
            .scan_parquet(work / indexed.INTERVAL_CHUNKS_NAME)
            .filter(pl.col('chunk_id').is_in(list(ids)))
            .join(pl.scan_parquet(work / indexed.KEY_INTERVALS_NAME), on='color')
            .group_by('chunk_id')
            .agg(
                n_intervals=pl.len(),
                n_hits=pl.col('hits').sum(),
                rank_lo=pl.col('left').min(),
                rank_hi=pl.col('right').max(),
            )
            .sort('chunk_id')
            .collect()
        )
        finished_ids = frozenset(
            filter(
                lambda chunk_i: session.store.parquet_valid(
                    chunk_path(chunk_i),
                    STAGE_SCHEMAS['parent_contrib'],
                ),
                map(int, census['chunk_id'].to_list()),
            )
        )

        def log_resume(chunk_i: int) -> None:
            row = census.filter(pl.col('chunk_id') == chunk_i)
            dest = chunk_path(chunk_i)
            _log.info(
                'patent_ate.score.stage',
                stage='parent_contrib_chunk',
                chunk_i=chunk_i,
                n_chunks=n_chunks,
                batch_i=batch_i,
                n_batches=n_batches,
                worker_i=worker_i,
                n_workers=n_workers,
                resumed=True,
                n_intervals=int(row['n_intervals'][0]),
                n_hits=int(row['n_hits'][0] or 0),
                rank_lo=int(row['rank_lo'][0] or 0),
                rank_hi=int(row['rank_hi'][0] or 0),
                wall_s=0.0,
                n_rows=int(pl.scan_parquet(dest).select(pl.len()).collect().item()),
                output_bytes=dest.stat().st_size,
                spill_bytes=0,
            )

        tuple(map(log_resume, sorted(finished_ids)))
        pending = census.filter(~pl.col('chunk_id').is_in(list(finished_ids)))
        if pending.is_empty():
            return

        pending_ids = tuple(map(int, pending['chunk_id'].to_list()))

        def census_ints(chunk_i: int) -> tuple[int, int, int, int]:
            row = pending.filter(pl.col('chunk_id') == chunk_i)
            return (
                int(row['n_intervals'][0]),
                int(row['n_hits'][0] or 0),
                int(row['rank_lo'][0] or 0),
                int(row['rank_hi'][0] or 0),
            )

        band_lo = int(pending.select(pl.col('rank_lo').min()).item() or 0)
        band_hi = int(pending.select(pl.col('rank_hi').max()).item() or 0)
        stats_path = session.store.stage_parquet('term_stats')

        def write(connection: Backend) -> int:
            batch_started = time.perf_counter()
            stats = connection.read_parquet(stats_path)
            self.replace_temp_table(connection, '_term_stats', stats)
            self.replace_temp_table(
                connection,
                '_identity',
                connection.read_parquet(work / indexed.IDENTITY_NAME),
            )
            self.replace_temp_table(
                connection,
                '_batch_intervals',
                connection.read_parquet(work / indexed.KEY_INTERVALS_NAME).inner_join(
                    connection.read_parquet(work / indexed.INTERVAL_CHUNKS_NAME).filter(
                        ibis._.chunk_id.isin(list(pending_ids))
                    ),
                    'color',
                ),
            )
            self.replace_temp_table(
                connection,
                '_sa_color_band',
                connection.read_parquet(work / indexed.SA_COLOR_NAME).filter(
                    (ibis._.rank >= band_lo) & (ibis._.rank < band_hi)
                ),
            )
            _log.info(
                'patent_ate.score.stage',
                stage='parent_contrib_batch',
                phase='start',
                batch_i=batch_i,
                n_batches=n_batches,
                worker_i=worker_i,
                n_workers=n_workers,
                n_packs=len(pending_ids),
                band_lo=band_lo,
                band_hi=band_hi,
                n_chunks=n_chunks,
            )

            def write_one(chunk_i: int) -> int:
                started = time.perf_counter()
                n_intervals, n_hits, rank_lo, rank_hi = census_ints(chunk_i)
                intervals = connection.table('_batch_intervals').filter(ibis._.chunk_id == chunk_i)
                band = connection.table('_sa_color_band').filter(
                    (ibis._.rank >= rank_lo) & (ibis._.rank < rank_hi)
                )
                raw = algebra.indexed_interval_pairs(
                    connection.table('_identity'),
                    intervals,
                    band,
                    rank_lo=rank_lo,
                    rank_hi=rank_hi,
                )
                stats_table = connection.table('_term_stats')
                n_rows, output_bytes, spill = self.publish_parquet(
                    connection,
                    algebra.compact_parent_pairs(
                        stats_table,
                        algebra.verified_parent_pairs(stats_table, raw),
                    ),
                    CONTRIB_CAST,
                    chunk_path(chunk_i),
                )
                _log.info(
                    'patent_ate.score.stage',
                    stage='parent_contrib_chunk',
                    chunk_i=chunk_i,
                    n_chunks=n_chunks,
                    batch_i=batch_i,
                    n_batches=n_batches,
                    worker_i=worker_i,
                    n_workers=n_workers,
                    resumed=False,
                    n_intervals=n_intervals,
                    n_hits=n_hits,
                    rank_lo=rank_lo,
                    rank_hi=rank_hi,
                    wall_s=round(time.perf_counter() - started, 6),
                    n_rows=n_rows,
                    output_bytes=output_bytes,
                    spill_bytes=spill,
                )
                return n_rows

            counts = tuple(map(write_one, pending_ids))
            _log.info(
                'patent_ate.score.stage',
                stage='parent_contrib_batch',
                phase='done',
                batch_i=batch_i,
                n_batches=n_batches,
                worker_i=worker_i,
                n_workers=n_workers,
                n_packs=len(pending_ids),
                wall_s=round(time.perf_counter() - batch_started, 6),
                n_rows=sum(counts),
            )
            return sum(counts)

        _ = self.run(write)

    def write_indexed_contributions(self, session: ScoreStageSession) -> None:  # ruff: ignore[complex-structure]
        """Score verified indexed pairs as resumable chunk contributions, then merge."""
        if self._ate.indexed is None:
            raise indexed.IndexedContainmentError('indexed containment requires indexed settings')
        work = session.store.indexed_dir()
        identity = indexed.term_identity(pl.read_parquet(session.store.stage_parquet('term_stats')))
        (work / 'short_candidates.parquet').unlink(missing_ok=True)
        (work / 'long_candidates.parquet').unlink(missing_ok=True)
        (work / 'candidate_pairs.parquet').unlink(missing_ok=True)
        (work / 'pattern_parents.parquet').unlink(missing_ok=True)
        maps_resumed = indexed.resume_or_write_pattern_parents(
            identity, work / 'pattern_parents.parquet'
        )
        n_chunks = INDEXED_CONTRIB_CHUNKS
        chunk_dir = session.store.bind_indexed_contrib_chunks(
            work, IndexedMapsFingerprint.from_maps(work, n_chunks)
        )

        def is_finished_chunk(path: Path) -> bool:
            return session.store.parquet_valid(path, STAGE_SCHEMAS['parent_contrib'])

        finished = tuple(
            map(  # ruff: ignore[unnecessary-map]
                lambda path: int(path.stem),
                filter(is_finished_chunk, sorted(chunk_dir.glob('*.parquet'))),
            )
        )
        ids = indexed.write_interval_chunks(
            work / indexed.KEY_INTERVALS_NAME,
            work / indexed.INTERVAL_CHUNKS_NAME,
            finished_ids=finished,
            max_intervals=INDEXED_MAX_INTERVALS_PER_CHUNK,
            max_cost=INDEXED_MAX_CHUNK_COST,
        )
        n_planned = len(ids)
        batches = tuple(batched(ids, INDEXED_CONTRIB_BATCH_PACKS))
        n_batches = len(batches)
        counted = getattr(os, 'process_cpu_count', os.cpu_count)
        host_cpus = max(1, int(counted() or os.cpu_count() or 1))
        n_workers = min(INDEXED_CONTRIB_BATCH_WORKERS, max(1, n_batches), host_cpus)
        threads_each = max(1, host_cpus // n_workers)

        def share_memory(limit: str, parts: int) -> str:
            match = re.fullmatch(r'(\d+)([KMGTP]?B)', limit.replace(' ', ''), flags=re.IGNORECASE)
            if match is None or parts <= 1:
                return limit
            return f'{max(1, int(match.group(1)) // parts)}{match.group(2)}'

        mem_each = share_memory(self._ate.duckdb_memory, n_workers)
        _log.info(
            'patent_ate.score.stage',
            stage='parent_contrib_chunks',
            maps_resumed=maps_resumed,
            n_chunks=n_chunks,
            n_chunk_ids=n_planned,
            n_finished=len(finished),
            n_batches=n_batches,
            batch_packs=INDEXED_CONTRIB_BATCH_PACKS,
            n_workers=n_workers,
            duckdb_threads=threads_each,
            duckdb_memory=mem_each,
            max_intervals=INDEXED_MAX_INTERVALS_PER_CHUNK,
            max_cost=INDEXED_MAX_CHUNK_COST,
        )

        def write_lane(
            lane: tuple[int, tuple[tuple[int, tuple[int, ...]], ...]],
        ) -> None:
            worker_i, lane_batches = lane
            worker = ScoreStageExecutor(
                self._ate.model_copy(
                    update={
                        'duckdb_threads': threads_each,
                        'duckdb_memory': mem_each,
                    }
                ),
                self._temp_dir / f'contrib-worker-{worker_i}',
            )

            def write_batch(indexed_batch: tuple[int, tuple[int, ...]]) -> None:
                batch_i, chunk_ids = indexed_batch
                worker.write_indexed_chunk_batch(
                    session,
                    work,
                    chunk_dir,
                    chunk_ids,
                    n_planned,
                    batch_i=batch_i,
                    n_batches=n_batches,
                    worker_i=worker_i,
                    n_workers=n_workers,
                )

            tuple(map(write_batch, lane_batches))

        def lane_for(
            worker_i: int,
        ) -> tuple[int, tuple[tuple[int, tuple[int, ...]], ...]]:
            return (
                worker_i,
                tuple(
                    map(  # ruff: ignore[unnecessary-map]
                        lambda batch_i: (batch_i, batches[batch_i]),
                        range(worker_i, n_batches, n_workers),
                    )
                ),
            )

        lanes = tuple(filter(itemgetter(1), map(lane_for, range(n_workers))))

        def run_lanes() -> None:
            if len(lanes) <= 1:
                tuple(map(write_lane, lanes))
                return
            with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
                futures = tuple(
                    map(lambda lane: pool.submit(write_lane, lane), lanes)  # ruff: ignore[unnecessary-map]
                )
                tuple(map(methodcaller('result'), futures))

        run_lanes()

        def chunk_path(chunk_i: int) -> Path:
            return chunk_dir / f'{chunk_i:05d}.parquet'

        parts = tuple(map(chunk_path, ids))
        _log.info(
            'patent_ate.score.stage',
            stage='parent_contrib_merge',
            n_files=len(parts),
            n_chunk_ids=n_planned,
        )
        self.materialize_stage(
            session,
            'parent_contrib',
            lambda connection: self.merge_contribution_parts(connection, parts),
            resume=False,
        )

    def write_scored(self, session: ScoreStageSession) -> None:
        """Score compact term stats against parent counts in hash buckets."""
        if session.stage_valid('scored') and 'scored' in session.completed:
            session.note_resume('scored')
            session.finish('scored')
            return
        n_buckets = SCORE_TERM_BUCKETS
        part_dir = session.store.stages_dir / SCORED_CHUNK_DIRNAME
        part_dir.mkdir(parents=True, exist_ok=True)
        stats_path = session.store.stage_parquet('term_stats')
        contrib_path = session.store.stage_parquet('parent_contrib')
        n_stats = int(pl.scan_parquet(stats_path).select(pl.len()).collect().item())
        n_contrib = int(pl.scan_parquet(contrib_path).select(pl.len()).collect().item())
        _log.info(
            'patent_ate.score.stage',
            stage='scored',
            phase='start',
            n_buckets=n_buckets,
            n_stats=n_stats,
            n_contrib=n_contrib,
        )

        def write_bucket(bucket_i: int) -> None:
            part = part_dir / f'{bucket_i:05d}.parquet'
            if session.store.parquet_valid(part, STAGE_SCHEMAS['scored']):
                _log.info(
                    'patent_ate.score.stage',
                    stage='scored_chunk',
                    bucket_i=bucket_i,
                    n_buckets=n_buckets,
                    resumed=True,
                    wall_s=0.0,
                    n_rows=int(pl.scan_parquet(part).select(pl.len()).collect().item()),
                    output_bytes=part.stat().st_size,
                    spill_bytes=0,
                )
                return

            def write(connection: Backend) -> int:
                started = time.perf_counter()
                stats = connection.read_parquet(stats_path)
                contrib = connection.read_parquet(contrib_path)
                n_rows, output_bytes, spill = self.publish_parquet(
                    connection,
                    algebra.scored_terms(
                        stats.filter(algebra.parent_bucket(stats.term, n_buckets) == bucket_i),
                        contrib.filter(algebra.parent_bucket(contrib.child, n_buckets) == bucket_i),
                    ),
                    SCORED_CAST,
                    part,
                )
                _log.info(
                    'patent_ate.score.stage',
                    stage='scored_chunk',
                    bucket_i=bucket_i,
                    n_buckets=n_buckets,
                    resumed=False,
                    wall_s=round(time.perf_counter() - started, 6),
                    n_rows=n_rows,
                    output_bytes=output_bytes,
                    spill_bytes=spill,
                )
                return n_rows

            _ = self.run(write)

        tuple(map(write_bucket, range(n_buckets)))
        parts = tuple(sorted(part_dir.glob('*.parquet')))
        _log.info(
            'patent_ate.score.stage',
            stage='scored_merge',
            n_files=len(parts),
            n_buckets=n_buckets,
        )
        self.materialize_stage(
            session,
            'scored',
            lambda connection: (
                connection.read_parquet(parts)
                if parts
                else ibis.memtable(
                    {'term': [], 'df': [], 'c_value': []},
                    schema={'term': 'string', 'df': 'int64', 'c_value': 'float64'},
                )
            ),
            resume=False,
        )

    def write_units(self, session: ScoreStageSession, plan: ParentWorkPlan) -> None:
        """Reuse valid unit files and write the rest in plan order."""
        ids = tuple(map(attrgetter('unit_id'), plan.units))
        kept = frozenset(
            filter(
                lambda unit_id: (
                    unit_id in session.finished_units and session.store.unit_valid(unit_id)
                ),
                ids,
            )
        )
        session.finished_units[:] = list(filter(kept.__contains__, ids))

        def write_or_resume(unit: ParentWorkUnit) -> None:
            if unit.unit_id in session.finished_units:
                session.note_resume(
                    f'parent_unit_{unit.unit_id}',
                    path=session.store.unit_path(unit.unit_id),
                )
                return
            self.write_unit(session, unit, plan)

        tuple(map(write_or_resume, plan.units))

    def commit_keys(self, session: ScoreStageSession, dest: Path) -> Path:
        """Write normalized keys and commit the immutable termhood generation."""

        def write(connection: Backend) -> Path:
            started = time.perf_counter()
            scored_path = session.store.stage_parquet('scored')
            n_scored = int(pl.scan_parquet(scored_path).select(pl.len()).collect().item())
            _log.info(
                'patent_ate.score.stage',
                stage='keys',
                phase='start',
                n_scored=n_scored,
            )
            partial = dest / f'{TERMHOOD_PARQUET_NAME}.partial'
            partial.unlink(missing_ok=True)
            _ = self.write_table(
                connection,
                algebra.cvalue_keys(
                    connection.read_parquet(scored_path),
                    connection.read_parquet(session.store.stage_parquet('surfaces')),
                ),
                KEYS_CAST,
                partial,
            )
            store = TermhoodStore.commit(partial, dest, total_docs=session.total_docs)
            _log.info(
                'patent_ate.score.stage',
                stage='keys',
                resumed=False,
                wall_s=round(time.perf_counter() - started, 6),
                n_rows=store.meta.n_keys,
                output_bytes=store.parquet.stat().st_size,
                spill_bytes=self.spill_bytes(),
            )
            _log.info(
                'patent_ate.score.cvalue',
                n_terms=store.meta.n_keys,
                n_docs=session.total_docs,
            )
            return Path(store.root)

        return self.run(write)

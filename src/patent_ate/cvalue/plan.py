"""Parent work-plan schemas and length-aware cost algebra."""

from __future__ import annotations

from operator import attrgetter
from typing import Annotated, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from patent_ate.spec import AteSpec

PARENT_UNIT_DIRNAME = 'parent_units'
PARENT_PLAN_NAME = 'parent_work_plan.json'
PARENT_SPAN_COSTS_NAME = 'parent_span_costs'
INDEXED_PLAN_NAME = 'indexed_containment_plan.json'
INDEXED_ARTIFACT_VERSION = '3'
INDEXED_WORKDIR = 'indexed'
INDEXED_CONTRIB_CHUNKS = 256
INDEXED_MAX_INTERVALS_PER_CHUNK = 50_000
INDEXED_MAX_CHUNK_COST = 1_500_000_000
INDEXED_CONTRIB_BATCH_PACKS = 64
INDEXED_CONTRIB_BATCH_WORKERS = 8
SCORE_TERM_BUCKETS = 8
INDEXED_CONTRIB_DIRNAME = 'contrib_chunks'
INDEXED_CONTRIB_MANIFEST = 'contrib_chunks.json'
SCORED_CHUNK_DIRNAME = 'scored_chunks'


class ParentWorkPlanError(ValueError):
    """Raised when a work plan exceeds a declared unit, scan, or base-slot cap."""


class HashSlotUnit(BaseModel):
    """Declared hash-range parent subset whose remaining mass is under the base caps."""

    model_config = ConfigDict(frozen=True)

    kind: Literal['hash_slot'] = 'hash_slot'
    unit_id: str = Field(min_length=1)
    total_slots: int = Field(ge=1)
    slot: int = Field(ge=0)
    estimated_windows: int = Field(ge=0)
    estimated_weighted_bytes: int = Field(ge=0)
    candidate_scan: Literal['full'] = 'full'


class SpanBandUnit(BaseModel):
    """Exact parent filter plus a candidate span-length band under the tail caps."""

    model_config = ConfigDict(frozen=True)

    kind: Literal['span_band'] = 'span_band'
    unit_id: str = Field(min_length=1)
    filter_artifact: str = Field(min_length=1)
    span_n_min: int = Field(ge=1)
    span_n_max: int = Field(ge=1)
    estimated_windows: int = Field(ge=0)
    estimated_weighted_bytes: int = Field(ge=0)
    estimated_candidate_rows: int = Field(ge=0)
    estimated_candidate_bytes: int = Field(ge=0)
    estimated_comparisons: int = Field(ge=0)
    strategy: Literal['window', 'candidate']
    candidate_scan: Literal['band'] = 'band'


ParentWorkUnit = Annotated[HashSlotUnit | SpanBandUnit, Field(discriminator='kind')]


class IndexedContainmentPlan(BaseModel):
    """Frozen identity of one indexed candidate-generation run."""

    model_config = ConfigDict(frozen=True)

    containment: Literal['indexed'] = 'indexed'
    artifact_version: str = Field(min_length=1)
    semantic_version: str = Field(min_length=1)
    engine: str = Field(min_length=1)
    prefix_bytes: int = Field(ge=1)
    n_terms: int = Field(ge=0)

    def matches(self, ate: AteSpec, *, semantic_version: str) -> bool:
        """Return whether this plan still matches the current indexed identity."""
        spec = ate.indexed
        if ate.containment != 'indexed' or spec is None:
            return False
        return (
            self.artifact_version == INDEXED_ARTIFACT_VERSION
            and self.semantic_version == semantic_version
            and self.engine == spec.engine
            and self.prefix_bytes == spec.prefix_bytes
        )


class ParentWorkPlan(BaseModel):
    """Deterministic parent work units whose preflight cost is the executed algebra."""

    model_config = ConfigDict(frozen=True)

    parent_buckets: int = Field(ge=1)
    tail_window_cap: int = Field(ge=1)
    tail_weighted_bytes_cap: int = Field(ge=1)
    base_window_cap: int = Field(ge=1)
    base_weighted_bytes_cap: int = Field(ge=1)
    max_parent_units: int = Field(ge=1)
    max_candidate_scans: int = Field(ge=1)
    tail_candidate_row_cap: int = Field(ge=1)
    tail_compare_cap: int = Field(ge=1)
    full_candidate_scans: int = Field(ge=0)
    band_filtered_scans: int = Field(ge=0)
    units: tuple[ParentWorkUnit, ...] = ()
    over_cap_filter: str | None = None

    def unit_ids(self) -> tuple[str, ...]:
        """Return unit identifiers in plan order."""
        return tuple(map(attrgetter('unit_id'), self.units))

    def matches(self, ate: AteSpec) -> bool:
        """Return whether this plan still matches ``ate`` caps and slot layout."""

        def hash_ok(unit: ParentWorkUnit) -> bool:
            return not isinstance(unit, HashSlotUnit) or (
                unit.total_slots == ate.parent_buckets
                and unit.estimated_windows <= ate.base_window_cap
                and unit.estimated_weighted_bytes <= ate.base_weighted_bytes_cap
            )

        def tail_ok(unit: ParentWorkUnit) -> bool:
            if not isinstance(unit, SpanBandUnit):
                return True
            match unit.strategy:
                case 'candidate':
                    candidate_ok = (
                        unit.estimated_candidate_rows <= ate.tail_candidate_row_cap
                        and unit.estimated_comparisons <= ate.tail_compare_cap
                    )
                    return bool(candidate_ok)
                case 'window':
                    window_ok = (
                        unit.estimated_windows <= ate.tail_window_cap
                        and unit.estimated_weighted_bytes <= ate.tail_weighted_bytes_cap
                    )
                    return bool(window_ok)

        return (
            self.parent_buckets == ate.parent_buckets
            and self.tail_window_cap == ate.tail_window_cap
            and self.tail_weighted_bytes_cap == ate.tail_weighted_bytes_cap
            and self.base_window_cap == ate.base_window_cap
            and self.base_weighted_bytes_cap == ate.base_weighted_bytes_cap
            and self.max_parent_units == ate.max_parent_units
            and self.max_candidate_scans == ate.max_candidate_scans
            and self.tail_candidate_row_cap == ate.tail_candidate_row_cap
            and self.tail_compare_cap == ate.tail_compare_cap
            and len(self.units) <= ate.max_parent_units
            and self.full_candidate_scans + self.band_filtered_scans <= ate.max_candidate_scans
            and all(map(hash_ok, self.units))
            and all(map(tail_ok, self.units))
        )


def reach_band_starts(frontier: pl.DataFrame, nxt: pl.DataFrame) -> pl.DataFrame:
    """Follow left-maximal pack cursors until every parent is exhausted."""
    if frontier.is_empty():
        return frontier
    stepped = (
        frontier
        .join(nxt, on=['term', 'cursor'])
        .select('term', cursor=pl.col('nxt'), n_len=pl.col('n_len'))
        .filter(pl.col('cursor') < pl.col('n_len'))
        .select('term', 'cursor')
    )
    if stepped.is_empty():
        return frontier
    return pl.concat([frontier, reach_band_starts(stepped, nxt)]).unique(['term', 'cursor'])


def peel_span_bands(
    over: pl.DataFrame,
    span_values: tuple[int, ...],
    length_census: pl.DataFrame,
    ate: AteSpec,
) -> pl.DataFrame:
    """Pack over-cap parents into tail-capped span-band rows."""
    empty = pl.DataFrame(
        schema={
            'kind': pl.String,
            'unit_id': pl.String,
            'filter_artifact': pl.String,
            'span_n_min': pl.Int64,
            'span_n_max': pl.Int64,
            'estimated_windows': pl.Int64,
            'estimated_weighted_bytes': pl.Int64,
            'estimated_candidate_rows': pl.Int64,
            'estimated_candidate_bytes': pl.Int64,
            'estimated_comparisons': pl.Int64,
            'strategy': pl.String,
            'candidate_scan': pl.String,
            'term': pl.String,
        }
    )
    if over.is_empty() or not span_values:
        return empty
    lengths = (
        over
        .select('term', 'parent_span_n', 'nbytes')
        .join(
            pl.DataFrame({'span_n': list(span_values)}, schema={'span_n': pl.Int64}),
            how='cross',
        )
        .filter(pl.col('span_n') < pl.col('parent_span_n'))
        .with_columns(
            starts=pl.col('parent_span_n') - pl.col('span_n') + 1,
        )
        .with_columns(
            weighted=(
                (
                    pl.col('starts').cast(pl.Float64)
                    * pl.col('nbytes')
                    * pl.col('span_n')
                    / pl.col('parent_span_n')
                )
                .ceil()
                .cast(pl.Int64)
            ),
        )
        .sort(['term', 'span_n'], descending=[False, True])
        .with_columns(pack_i=pl.int_range(0, pl.len()).over('term'))
    )
    if lengths.is_empty():
        return empty
    atomic = lengths.filter(
        (pl.col('starts') > ate.tail_window_cap)
        | (pl.col('weighted') > ate.tail_weighted_bytes_cap)
    )
    if atomic.height:
        raise ParentWorkPlanError('atomic parent length exceeds the declared tail caps')
    ranked = lengths.with_columns(
        cum_w=pl.col('starts').cum_sum().over('term'),
        cum_b=pl.col('weighted').cum_sum().over('term'),
        n_len=pl.len().over('term'),
    )
    origins = ranked.select(
        'term',
        start=pl.col('pack_i'),
        origin_w=pl.col('cum_w') - pl.col('starts'),
        origin_b=pl.col('cum_b') - pl.col('weighted'),
        n_len=pl.col('n_len'),
    )
    extents = ranked.select(
        'term',
        end=pl.col('pack_i'),
        end_w=pl.col('cum_w'),
        end_b=pl.col('cum_b'),
    )
    valid = (
        origins
        .join(extents, on='term')
        .filter(pl.col('end') >= pl.col('start'))
        .with_columns(
            windows=pl.col('end_w') - pl.col('origin_w'),
            weighted_bytes=pl.col('end_b') - pl.col('origin_b'),
        )
        .filter(
            (pl.col('windows') <= ate.tail_window_cap)
            & (pl.col('weighted_bytes') <= ate.tail_weighted_bytes_cap)
        )
    )
    farthest = valid.group_by('term', 'start').agg(
        farthest_end=pl.col('end').max(),
        n_len=pl.col('n_len').first(),
    )
    if farthest.height != lengths.height:
        raise ParentWorkPlanError('length band exceeds the declared tail caps')
    nxt = farthest.select(
        'term',
        cursor=pl.col('start'),
        nxt=pl.col('farthest_end') + 1,
        n_len=pl.col('n_len'),
    )
    seed = (
        lengths
        .select('term')
        .unique()
        .select(
            'term',
            cursor=pl.lit(0, dtype=pl.Int64),
        )
    )
    packed = (
        reach_band_starts(seed, nxt)
        .join(farthest, left_on=['term', 'cursor'], right_on=['term', 'start'])
        .select('term', start=pl.col('cursor'), end=pl.col('farthest_end'))
        .join(
            valid.select('term', 'start', 'end', 'windows', 'weighted_bytes'),
            on=['term', 'start', 'end'],
        )
        .join(ranked.select('term', 'pack_i', 'span_n'), on='term')
        .filter((pl.col('pack_i') >= pl.col('start')) & (pl.col('pack_i') <= pl.col('end')))
        .group_by('term', 'start', 'end', 'windows', 'weighted_bytes')
        .agg(
            span_n_min=pl.col('span_n').min(),
            span_n_max=pl.col('span_n').max(),
        )
        .with_columns(span_n=pl.int_ranges(pl.col('span_n_min'), pl.col('span_n_max') + 1))
        .explode('span_n')
        .join(length_census, on='span_n', how='left')
        .with_columns(
            pl.col('candidate_rows').fill_null(0),
            pl.col('candidate_bytes').fill_null(0),
        )
        .group_by('term', 'start', 'span_n_min', 'span_n_max', 'windows', 'weighted_bytes')
        .agg(
            estimated_candidate_rows=pl.col('candidate_rows').sum(),
            estimated_candidate_bytes=pl.col('candidate_bytes').sum(),
        )
        .with_columns(estimated_comparisons=pl.col('estimated_candidate_rows'))
    )
    under_candidate = (pl.col('estimated_candidate_rows') <= ate.tail_candidate_row_cap) & (
        pl.col('estimated_comparisons') <= ate.tail_compare_cap
    )
    chosen = packed.with_columns(
        strategy=pl
        .when((pl.col('estimated_comparisons') < pl.col('windows')) & under_candidate)
        .then(pl.lit('candidate'))
        .when(
            (pl.col('windows') <= ate.tail_window_cap)
            & (pl.col('weighted_bytes') <= ate.tail_weighted_bytes_cap)
        )
        .then(pl.lit('window'))
        .otherwise(pl.lit('overflow'))
    )
    if chosen.filter(pl.col('strategy') == 'overflow').height:
        raise ParentWorkPlanError('length band exceeds the declared tail caps')
    return (
        chosen
        .sort(['term', 'start'])
        .with_row_index('span_ordinal', offset=1)
        .select(
            kind=pl.lit('span_band'),
            unit_id=pl.format(
                'span-{}',
                pl.col('span_ordinal').cast(pl.String).str.zfill(4),
            ),
            filter_artifact=pl.format(
                f'{PARENT_UNIT_DIRNAME}/filters/{{}}',
                pl.format(
                    'span-{}',
                    pl.col('span_ordinal').cast(pl.String).str.zfill(4),
                )
                + pl.lit('.parquet'),
            ),
            span_n_min=pl.col('span_n_min'),
            span_n_max=pl.col('span_n_max'),
            estimated_windows=pl.col('windows'),
            estimated_weighted_bytes=pl.col('weighted_bytes'),
            estimated_candidate_rows=pl.col('estimated_candidate_rows'),
            estimated_candidate_bytes=pl.col('estimated_candidate_bytes'),
            estimated_comparisons=pl.col('estimated_comparisons'),
            strategy=pl.col('strategy'),
            candidate_scan=pl.lit('band'),
            term=pl.col('term'),
        )
    )


def hash_slot_units(under: pl.DataFrame, ate: AteSpec) -> pl.DataFrame:
    """Aggregate remaining parents into declared hash-slot rows."""
    empty = pl.DataFrame(
        schema={
            'kind': pl.String,
            'unit_id': pl.String,
            'total_slots': pl.Int64,
            'slot': pl.Int64,
            'estimated_windows': pl.Int64,
            'estimated_weighted_bytes': pl.Int64,
            'candidate_scan': pl.String,
        }
    )
    if under.is_empty():
        return empty
    slots = (
        under
        .group_by('slot')
        .agg(
            estimated_windows=pl.col('estimated_windows').sum(),
            estimated_weighted_bytes=pl.col('estimated_weighted_bytes').sum(),
        )
        .sort('slot')
    )
    overflow = slots.filter(
        (pl.col('estimated_windows') > ate.base_window_cap)
        | (pl.col('estimated_weighted_bytes') > ate.base_weighted_bytes_cap)
    )
    if overflow.height:
        raise ParentWorkPlanError(
            'declared hash slot exceeds the base aggregate caps after peeling'
        )
    return slots.select(
        kind=pl.lit('hash_slot'),
        unit_id=pl.format('hash-{}-{}', pl.lit(ate.parent_buckets), pl.col('slot')),
        total_slots=pl.lit(ate.parent_buckets, dtype=pl.Int64),
        slot=pl.col('slot'),
        estimated_windows=pl.col('estimated_windows'),
        estimated_weighted_bytes=pl.col('estimated_weighted_bytes'),
        candidate_scan=pl.lit('full'),
    )


def build_parent_work_plan(
    costs: pl.DataFrame,
    span_values: tuple[int, ...],
    *,
    ate: AteSpec,
    census: pl.DataFrame | None = None,
) -> tuple[ParentWorkPlan, dict[str, tuple[str, ...]]]:
    """Peel over-cap parents into length bands and keep declared hash slots.

    ``costs`` is the already-materialized per-parent frame, including the
    declared hash slot. ``parent_buckets`` stays as given. ``census`` is
    per-``span_n`` candidate rows and bytes; when omitted, those counts
    come from ``costs``.
    """

    def singleton_term(term: object) -> tuple[str, ...]:
        return (str(term),)

    length_census = (
        census
        if census is not None
        else (
            costs.group_by(span_n=pl.col('parent_span_n')).agg(
                candidate_rows=pl.len(),
                candidate_bytes=pl.col('nbytes').sum(),
            )
        )
    )
    over = costs.filter(
        (pl.col('estimated_windows') > ate.tail_window_cap)
        | (pl.col('estimated_weighted_bytes') > ate.tail_weighted_bytes_cap)
    )
    under = costs.filter(
        (pl.col('estimated_windows') <= ate.tail_window_cap)
        & (pl.col('estimated_weighted_bytes') <= ate.tail_weighted_bytes_cap)
    )
    span_frame = peel_span_bands(over, span_values, length_census, ate)
    hash_frame = hash_slot_units(under, ate)
    span_units = tuple(map(SpanBandUnit.model_validate, span_frame.to_dicts()))
    hash_units = tuple(map(HashSlotUnit.model_validate, hash_frame.to_dicts()))
    mixed: tuple[ParentWorkUnit, ...] = (*span_units, *hash_units)
    units: tuple[ParentWorkUnit, ...] = tuple(sorted(mixed, key=attrgetter('unit_id')))
    full_scans = hash_frame.height
    band_scans = span_frame.height
    if len(units) > ate.max_parent_units:
        raise ParentWorkPlanError('parent work-unit count exceeds max_parent_units')
    if full_scans + band_scans > ate.max_candidate_scans:
        raise ParentWorkPlanError('candidate scans exceed max_candidate_scans')
    over_rel = f'{PARENT_UNIT_DIRNAME}/filters/over_cap_parents.parquet' if over.height else None
    over_terms = tuple(sorted(map(str, over['term'].to_list()))) if over.height else ()
    span_filters = dict(
        zip(
            span_frame['filter_artifact'].to_list(),
            map(singleton_term, span_frame['term'].to_list()),
            strict=True,
        )
    )
    filters = {**span_filters, **({over_rel: over_terms} if over_rel is not None else {})}
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
        full_candidate_scans=full_scans,
        band_filtered_scans=band_scans,
        units=units,
        over_cap_filter=over_rel,
    )
    return plan, filters

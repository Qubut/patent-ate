"""Indexed containment over a concatenated-key suffix array.

Construction is one ``divsufsort`` call. Coarse bounds are a
``searchsorted`` pair on SA-ordered eight-byte prefixes. Longer keys are
refined to the true key length inside each unique coarse interval.
Listing persists those intervals and one SA color column. Short keys use
the same array.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import numpy.typing as npt
import polars as pl
import structlog
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel, ConfigDict, Field
from pydivsufsort import divsufsort
from returns.io import impure_safe
from returns.pipeline import is_successful
from returns.unsafe import unsafe_perform_io

from patent_ate.spec import AteSpec

from .plan import (
    INDEXED_ARTIFACT_VERSION,
    INDEXED_MAX_CHUNK_COST,
    INDEXED_MAX_INTERVALS_PER_CHUNK,
    IndexedContainmentPlan,
)
from .store import ScoreStageSession, ScoreStageStore
from .text import surface_key

KEY_INTERVAL_SCHEMA = pl.Schema({
    'color': pl.Int64,
    'key': pl.String,
    'left': pl.Int64,
    'right': pl.Int64,
})
SA_COLOR_SCHEMA = pl.Schema({'rank': pl.Int64, 'parent_color': pl.Int32})
IDENTITY_SCHEMA = pl.Schema({
    'color': pl.Int64,
    'term': pl.String,
    'key': pl.String,
    'norm_nbytes': pl.Int64,
    'tier': pl.String,
})
INTERVAL_CHUNKS_SCHEMA = pl.Schema({
    'color': pl.Int64,
    'chunk_id': pl.Int64,
    'hits': pl.Int64,
})
IDENTITY_COLUMNS = ('color', 'term', 'key', 'norm_nbytes', 'tier')
KEY_INTERVALS_NAME = 'key_intervals.parquet'
SA_COLOR_NAME = 'sa_color.parquet'
IDENTITY_NAME = 'term_identity.parquet'
INTERVAL_CHUNKS_NAME = 'interval_chunks.parquet'
INT32_MAX = np.iinfo(np.int32).max
LISTING_RANK_BUDGET = 4_000_000
_log = structlog.get_logger(__name__)


class IndexedContainmentError(RuntimeError):
    """Raised when GSA construction or prefix listing is invalid."""


class PrefixBand(BaseModel):
    """Packed prefix width used as a uint64 search key."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(default=8, ge=8, le=8)
    exclusive_upper: int = Field(default=8, ge=8, le=8)


class ConcatenatedKeys(BaseModel):
    """NUL-separated UTF-8 of every normalized key, plus document starts."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: npt.NDArray[np.uint8]
    starts: npt.NDArray[np.int64]
    n_terms: int
    n: int


class GsaArrays(BaseModel):
    """Suffix array, eight-byte SA keys, and SA-ordered document colors."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    concat: ConcatenatedKeys
    sa: npt.NDArray[np.int32]
    sa_keys: npt.NDArray[np.uint64]
    sa_doc: npt.NDArray[np.int32]


class PrefixBounds(BaseModel):
    """Left and right SA ranks of every key's packed prefix."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    frame: pl.DataFrame


PREFIX = PrefixBand.model_validate({})


def term_identity(stats: pl.DataFrame) -> pl.DataFrame:
    """Assign one color per original term and the semantic-6 normalized key."""
    unique = stats.select('term').unique(maintain_order=False).sort('term')
    return (
        unique
        .with_columns(
            color=pl.int_range(0, unique.height, dtype=pl.Int64),
            key=pl.col('term').map_elements(
                lambda term: surface_key(str(term)),
                return_dtype=pl.String,
            ),
        )
        .with_columns(norm_nbytes=pl.col('key').str.len_bytes().cast(pl.Int64))
        .with_columns(
            tier=pl
            .when(pl.col('norm_nbytes') < PREFIX.exclusive_upper)
            .then(pl.lit('short'))
            .otherwise(pl.lit('long'))
        )
    )


def concat_key_text(identity: pl.DataFrame) -> ConcatenatedKeys:
    """Join key UTF-8 with a trailing NUL sentinel on every record."""
    bad = identity.filter(
        pl.col('key').eq('')
        | pl.col('key').str.len_bytes().eq(0)
        | pl.col('key').cast(pl.Binary).bin.contains(b'\x00')
    )
    if bad.height:
        raise IndexedContainmentError('normalized keys must be nonempty without NUL')
    if identity.height == 0:
        return ConcatenatedKeys(
            text=np.zeros(0, dtype=np.uint8),
            starts=np.zeros(0, dtype=np.int64),
            n_terms=0,
            n=0,
        )
    blob = str(identity.select(pl.col('key').implode().list.join('\u0000')).item())
    text = np.frombuffer((blob + '\u0000').encode('utf-8'), dtype=np.uint8).copy()
    if text.size > INT32_MAX:
        raise IndexedContainmentError('concatenated keys exceed the int32 suffix-array limit')
    widths = identity.select(width=pl.col('key').str.len_bytes() + 1)['width'].to_numpy()
    starts = np.cumsum(widths, dtype=np.int64) - widths.astype(np.int64)
    return ConcatenatedKeys(
        text=text,
        starts=starts,
        n_terms=identity.height,
        n=int(text.size),
    )


def build_gsa(concat: ConcatenatedKeys) -> GsaArrays:
    """Build the suffix array, eight-byte keys, and SA document colors."""

    def pack_prefix8(
        padded: npt.NDArray[np.uint8],
        pos: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.uint64]:
        packed = (
            padded[pos].astype(np.uint64) << 56
            | padded[pos + 1].astype(np.uint64) << 48
            | padded[pos + 2].astype(np.uint64) << 40
            | padded[pos + 3].astype(np.uint64) << 32
            | padded[pos + 4].astype(np.uint64) << 24
            | padded[pos + 5].astype(np.uint64) << 16
            | padded[pos + 6].astype(np.uint64) << 8
            | padded[pos + 7].astype(np.uint64)
        )
        return np.asarray(packed, dtype=np.uint64)

    if concat.n == 0:
        empty_sa = np.zeros(0, dtype=np.int32)
        return GsaArrays(
            concat=concat,
            sa=empty_sa,
            sa_keys=np.zeros(0, dtype=np.uint64),
            sa_doc=np.zeros(0, dtype=np.int32),
        )
    sa = np.ascontiguousarray(divsufsort(np.ascontiguousarray(concat.text)), dtype=np.int32)
    pad = np.zeros(PREFIX.width, dtype=np.uint8)
    padded = np.concatenate([concat.text, pad])
    sa_keys = pack_prefix8(padded, sa)
    positions = np.arange(concat.n, dtype=np.int64)
    doc = np.searchsorted(concat.starts, positions, side='right').astype(np.int64) - 1
    doc = np.where(concat.text == 0, np.int64(-1), doc)
    sa_doc = np.asarray(doc[sa], dtype=np.int32)
    return GsaArrays(concat=concat, sa=sa, sa_keys=sa_keys, sa_doc=sa_doc)


def prefix_bounds(identity: pl.DataFrame, arrays: GsaArrays) -> PrefixBounds:
    """Locate every key's eight-byte prefix interval with one batched searchsorted."""
    if identity.height == 0 or arrays.concat.n == 0:
        return PrefixBounds(frame=pl.DataFrame(schema=KEY_INTERVAL_SCHEMA))

    def prefix_word(pad: str) -> pl.Expr:
        return (
            pl
            .col('key')
            .cast(pl.Binary)
            .bin.slice(0, PREFIX.width)
            .bin.encode('hex')
            .str.pad_end(16, pad)
            .str.to_integer(base=16, dtype=pl.UInt64)
        )

    packed = identity.select(
        color=pl.col('color'),
        key=pl.col('key'),
        lo=prefix_word('0'),
        hi=prefix_word('f'),
    )
    left = np.searchsorted(
        arrays.sa_keys,
        packed['lo'].to_numpy().astype(np.uint64, copy=False),
        side='left',
    )
    right = np.searchsorted(
        arrays.sa_keys,
        packed['hi'].to_numpy().astype(np.uint64, copy=False),
        side='right',
    )
    return PrefixBounds(
        frame=packed.select('color', 'key').with_columns(
            left=pl.Series('left', left, dtype=pl.Int64),
            right=pl.Series('right', right, dtype=pl.Int64),
        )
    )


def exact_bounds(
    identity: pl.DataFrame,
    arrays: GsaArrays,
    coarse: PrefixBounds,
) -> PrefixBounds:
    """Tighten coarse eight-byte intervals to the true key length."""
    if coarse.frame.is_empty() or arrays.concat.n == 0:
        return PrefixBounds(frame=pl.DataFrame(schema=KEY_INTERVAL_SCHEMA))
    frame = coarse.frame.join(identity.select('color', 'norm_nbytes'), on='color')
    short = frame.filter(pl.col('norm_nbytes') <= PREFIX.width).select(
        'color', 'key', 'left', 'right'
    )
    long = frame.filter(pl.col('norm_nbytes') > PREFIX.width)
    if long.is_empty():
        return PrefixBounds(frame=short)
    max_n = int(long['norm_nbytes'].to_numpy().max())
    padded = np.concatenate([
        arrays.concat.text,
        np.zeros(max_n, dtype=np.uint8),
    ])
    ordered = long.sort(['left', 'right', 'norm_nbytes', 'color'])
    iv = np.stack(
        [ordered['left'].to_numpy(), ordered['right'].to_numpy()],
        axis=1,
    )
    _, group_starts = np.unique(iv, axis=0, return_index=True)
    group_starts = np.sort(np.asarray(group_starts, dtype=np.int64))
    group_stops = np.append(group_starts[1:], np.int64(ordered.height))
    origins = np.stack([group_starts, group_stops], axis=1)

    def refine_interval(origin: npt.NDArray[np.int64]) -> pl.DataFrame:
        piece = ordered.slice(int(origin[0]), int(origin[1]) - int(origin[0]))
        left = int(piece['left'][0])
        right = int(piece['right'][0])
        if right <= left:
            return piece.select('color', 'key', 'left', 'right')
        sa_pos = arrays.sa[left:right]
        lengths = piece['norm_nbytes'].unique().to_numpy()

        def refine_length(length: np.int64) -> pl.DataFrame:
            width = int(length)
            members = piece.filter(pl.col('norm_nbytes') == width)
            packed = np.ascontiguousarray(sliding_window_view(padded, width)[sa_pos])
            packed_keys = packed.view(np.dtype(f'S{width}')).reshape(-1)
            queries = np.asarray(
                members.select(pl.col('key').cast(pl.Binary)).to_series().to_list(),
                dtype=np.dtype(f'S{width}'),
            )
            new_left = np.int64(left) + np.searchsorted(packed_keys, queries, side='left')
            new_right = np.int64(left) + np.searchsorted(packed_keys, queries, side='right')
            return members.select('color', 'key').with_columns(
                left=pl.Series('left', new_left, dtype=pl.Int64),
                right=pl.Series('right', new_right, dtype=pl.Int64),
            )

        return pl.concat(tuple(map(refine_length, lengths)))

    refined = pl.concat(tuple(map(refine_interval, origins)))
    return PrefixBounds(frame=pl.concat([short, refined]))


def write_pattern_parents(identity: pl.DataFrame, dest: Path) -> Path:
    """Write identity, exact key intervals, and SA document colors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    identity_path = dest.parent / IDENTITY_NAME
    intervals_path = dest.parent / KEY_INTERVALS_NAME
    sa_color_path = dest.parent / SA_COLOR_NAME
    concat = concat_key_text(identity)
    arrays = build_gsa(concat)
    coarse = prefix_bounds(identity, arrays)
    arrays = arrays.model_copy(update={'sa_keys': np.zeros(0, dtype=np.uint64)})
    bounds = exact_bounds(identity, arrays, coarse)

    def persist_sa_color(sa_doc: npt.NDArray[np.int32], path: Path) -> Path:
        n = int(sa_doc.size)
        if n == 0:
            pl.DataFrame(schema=SA_COLOR_SCHEMA).write_parquet(path)
            return path
        starts = np.arange(0, n, LISTING_RANK_BUDGET, dtype=np.int64)

        def persist_batch(start: np.int64) -> Path:
            origin = int(start)
            stop = min(origin + LISTING_RANK_BUDGET, n)
            part = Path(chunk_dir) / f'{origin}.parquet'
            pl.DataFrame({
                'rank': np.arange(origin, stop, dtype=np.int64),
                'parent_color': sa_doc[origin:stop],
            }).write_parquet(part)
            return part

        with TemporaryDirectory(dir=path.parent, prefix=f'.{path.stem}.') as chunk_dir:
            parts = tuple(map(persist_batch, starts))
            (
                pl.scan_parquet(list(parts)).sink_parquet(
                    path, engine='streaming', maintain_order=False
                )
            )
        return path

    @impure_safe
    def persist() -> Path:
        identity.select(*IDENTITY_COLUMNS).write_parquet(identity_path)
        dest.unlink(missing_ok=True)
        if bounds.frame.is_empty():
            pl.DataFrame(schema=KEY_INTERVAL_SCHEMA).write_parquet(intervals_path)
            persist_sa_color(arrays.sa_doc, sa_color_path)
            return intervals_path
        open_bounds = bounds.frame.filter(pl.col('right') > pl.col('left'))
        bounds.frame.select('color', 'key', 'left', 'right').write_parquet(intervals_path)
        persist_sa_color(arrays.sa_doc, sa_color_path)
        _log.info(
            'patent_ate.score.stage',
            stage='pattern_parents',
            n_terms=identity.height,
            n_open=open_bounds.height,
            n_sa=int(arrays.sa_doc.size),
            n_intervals=bounds.frame.height,
        )
        return intervals_path

    result = persist()
    if not is_successful(result):
        raise unsafe_perform_io(result.failure())
    return unsafe_perform_io(result.unwrap())


def resume_or_write_pattern_parents(identity: pl.DataFrame, dest: Path) -> bool:
    """Reuse exact interval maps when they match the live term identity.

    Returns True when the three map files stay in place and listing is
    skipped. Returns False after a full write.
    """
    work = dest.parent
    identity_path = work / IDENTITY_NAME
    intervals_path = work / KEY_INTERVALS_NAME
    sa_color_path = work / SA_COLOR_NAME
    ready = (
        ScoreStageStore.parquet_valid(identity_path, IDENTITY_SCHEMA)
        and ScoreStageStore.parquet_valid(intervals_path, KEY_INTERVAL_SCHEMA)
        and ScoreStageStore.parquet_valid(sa_color_path, SA_COLOR_SCHEMA)
        and int(pl.scan_parquet(identity_path).select(pl.len()).collect().item()) == identity.height
        and int(pl.scan_parquet(intervals_path).select(pl.len()).collect().item())
        == identity.height
    )
    if not ready:
        write_pattern_parents(identity, dest)
        return False
    n_intervals = int(pl.scan_parquet(intervals_path).select(pl.len()).collect().item())
    n_open = int(
        pl
        .scan_parquet(intervals_path)
        .filter(pl.col('right') > pl.col('left'))
        .select(pl.len())
        .collect()
        .item()
    )
    n_sa = int(pl.scan_parquet(sa_color_path).select(pl.len()).collect().item())
    _log.info(
        'patent_ate.score.stage',
        stage='pattern_parents',
        resumed=True,
        n_terms=identity.height,
        n_open=n_open,
        n_sa=n_sa,
        n_intervals=n_intervals,
    )
    return True


def write_interval_chunks(  # ruff: ignore[complex-structure]
    intervals_path: Path,
    dest: Path,
    *,
    finished_ids: tuple[int, ...] = (),
    max_intervals: int = INDEXED_MAX_INTERVALS_PER_CHUNK,
    max_cost: int = INDEXED_MAX_CHUNK_COST,
) -> tuple[int, ...]:
    """Assign unfinished intervals to chunks bounded by PWMJ cost.

    Colors whose contribution file already exists keep their chunk id.
    Remaining intervals stay in left-then-color order. Contiguous runs are
    split until each run is under ``max_intervals`` and under ``max_cost``
    (that run's rank-band width times its interval count).
    """

    def assign_cost_bounded_chunk_ids(
        left: npt.NDArray[np.int64],
        right: npt.NDArray[np.int64],
        *,
        next_id: int,
    ) -> npt.NDArray[np.int64]:
        chunk_of = np.empty(left.shape[0], dtype=np.int64)

        def group_fits(lo: int, hi: int) -> bool:
            count = hi - lo
            band = int(right[lo:hi].max() - left[lo])
            return count <= max_intervals and count * max(band, 1) <= max_cost

        def paint(lo: int, hi: int, chunk_id: int) -> int:
            count = hi - lo
            if count <= 0:
                return chunk_id
            if count == 1 or group_fits(lo, hi):
                chunk_of[lo:hi] = chunk_id
                return chunk_id + 1
            mid = lo + count // 2
            chunk_id = paint(lo, mid, chunk_id)
            return paint(mid, hi, chunk_id)

        paint(0, left.shape[0], next_id)
        return chunk_of

    if max_intervals < 1:
        raise IndexedContainmentError('max_intervals must be at least 1')
    if max_cost < 1:
        raise IndexedContainmentError('max_cost must be at least 1')
    scanned = pl.scan_parquet(intervals_path)
    if int(scanned.select(pl.len()).collect().item()) == 0:
        pl.DataFrame(schema=INTERVAL_CHUNKS_SCHEMA).write_parquet(dest)
        return ()
    if finished_ids and not dest.is_file():
        raise IndexedContainmentError(
            'finished contribution chunks require interval_chunks.parquet'
        )
    base = scanned.with_columns(hits=(pl.col('right') - pl.col('left')).cast(pl.Int64)).select(
        'color', 'left', 'hits'
    )
    kept = (
        pl
        .scan_parquet(dest)
        .filter(pl.col('chunk_id').is_in(list(finished_ids)))
        .select('color', 'chunk_id', 'hits')
        if finished_ids
        else pl.LazyFrame(schema=INTERVAL_CHUNKS_SCHEMA)
    )
    pending = base.join(kept.select('color'), on='color', how='anti')
    next_id = (max(finished_ids) + 1) if finished_ids else 0
    n_pending = int(pending.select(pl.len()).collect().item())
    if n_pending:
        ordered = (
            pending
            .sort(['left', 'color'])
            .with_columns(right=pl.col('left') + pl.col('hits'))
            .collect()
        )
        chunk_of = assign_cost_bounded_chunk_ids(
            ordered['left'].to_numpy(),
            ordered['right'].to_numpy(),
            next_id=next_id,
        )
        packed = (
            ordered
            .select('color', 'hits')
            .with_columns(chunk_id=pl.Series('chunk_id', chunk_of))
            .select('color', 'chunk_id', 'hits')
            .lazy()
        )
    else:
        packed = pl.LazyFrame(schema=INTERVAL_CHUNKS_SCHEMA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f'{dest.name}.partial')
    partial.unlink(missing_ok=True)
    pl.concat([kept, packed], how='vertical').collect().write_parquet(partial)
    partial.replace(dest)
    return tuple(
        pl
        .scan_parquet(dest)
        .select('chunk_id')
        .unique()
        .sort('chunk_id')
        .collect()['chunk_id']
        .to_list()
    )


def build_indexed_plan(
    identity: pl.DataFrame,
    ate: AteSpec,
    *,
    semantic_version: str,
) -> IndexedContainmentPlan:
    """Freeze indexed identity counts and the GSA engine pin."""
    spec = ate.indexed
    if spec is None:
        raise IndexedContainmentError('indexed containment requires indexed settings')
    return IndexedContainmentPlan(
        artifact_version=INDEXED_ARTIFACT_VERSION,
        semantic_version=semantic_version,
        engine=spec.engine,
        prefix_bytes=spec.prefix_bytes,
        n_terms=identity.height,
    )


def resume_or_write_plan(
    session: ScoreStageSession,
    ate: AteSpec,
    *,
    semantic_version: str,
) -> tuple[IndexedContainmentPlan, bool]:
    """Reuse a matching indexed plan or publish a new one."""
    identity = term_identity(pl.read_parquet(session.store.stage_parquet('term_stats')))
    built = build_indexed_plan(identity, ate, semantic_version=semantic_version)
    reused = session.store.load_matching_indexed_plan(ate) if session.resume_units else None
    plan = reused if reused is not None else built
    if reused is None:
        if session.resume_units:
            session.store.wipe_units_and_scored()
        session.store.replace_indexed_plan(plan)
        session.finish('term_stats', ())
    return plan, session.resume_units and reused is not None

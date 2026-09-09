"""Ibis expressions for C-value: chunk lengths, parent windows, costs, and scores."""

from __future__ import annotations

import ibis
import structlog
from ibis.expr.types import IntegerValue, StringValue, Table

from patent_ate.termhood import TermhoodTable

from .text import (
    JATE_CHUNK_PATTERN,
    jate_contains,
    jate_word_count,
    normalized_key,
    regexp_extract_all,
)

_log = structlog.get_logger(__name__)


def parent_bucket(term: StringValue, buckets: int) -> IntegerValue:
    """Assign ``term`` to one hash-range partition in ``[0, buckets)``.

    The hash is a partition key only. Containment and joins still use the
    original term strings, so a hash collision cannot change scores.
    """
    width = ibis.literal(buckets, type='int64')
    hashed = term.hash().cast('int64')
    return ((hashed % width) + width) % width


def chunk_aligned(stats: Table) -> Table:
    """Attach Unicode word-run chunks and JATE whitespace counts to terms."""
    chunked = stats.mutate(
        chunks=regexp_extract_all(ibis._.term, JATE_CHUNK_PATTERN),
        word_count=jate_word_count(ibis._.term),
    ).mutate(chunk_n=ibis._.chunks.length().cast('int64'))
    return chunked.filter(chunked.chunk_n > 0)


def candidate_span_lengths(stats: Table) -> Table:
    """Return distinct chunker lengths present on candidate terms."""
    aligned = chunk_aligned(stats)
    return aligned.select(span_n=aligned.chunk_n).filter(ibis._.span_n > 0).distinct()


def candidate_span_census(stats: Table) -> Table:
    """Return candidate row and byte counts grouped by chunker length."""
    aligned = chunk_aligned(stats)
    return (
        aligned
        .group_by(span_n=aligned.chunk_n)
        .agg(
            candidate_rows=ibis._.count().cast('int64'),
            candidate_bytes=aligned.term.length().sum().cast('int64'),
        )
        .filter(ibis._.span_n > 0)
    )


def parent_windows(stats: Table, span_lengths: Table | None = None) -> Table:
    """Emit contiguous original-string slices of existing candidate lengths.

    One constant-pattern chunk scan per parent groups Unicode word runs and
    splits nonword codepoints. Each parent joins only candidate chunker
    lengths strictly below its own chunk count, then unnests every start
    that can produce that many chunks. Slices never begin inside a word
    run. Repeated rows collapse to one parent.
    """
    lengths = span_lengths if span_lengths is not None else candidate_span_lengths(stats)
    aligned = chunk_aligned(stats)
    joined = aligned.inner_join(lengths, lengths.span_n < aligned.chunk_n)
    started = joined.mutate(
        chunk_start=ibis.range(0, joined.chunk_n - joined.span_n + 1).unnest(),
    )
    generated = started.select(
        parent=started.term,
        parent_tf=started.tf,
        parent_word_count=started.word_count,
        window=started.chunks[started.chunk_start : started.chunk_start + started.span_n].join(
            '',
        ),
    )
    return generated.filter(generated.window.length() > 0).distinct()


def parent_span_costs(stats: Table, span_lengths: Table, *, buckets: int) -> Table:
    """Return per-parent executed window and weighted-byte estimates.

    The estimate is ``sum(parent_span_n - L + 1)`` over candidate chunker
    lengths ``L`` strictly below that parent. ``slot`` is the declared hash
    partition of the parent string.
    """
    aligned = chunk_aligned(stats.filter(stats.word_count >= 2))
    joined = aligned.inner_join(span_lengths, span_lengths.span_n < aligned.chunk_n)
    starts = (aligned.chunk_n - span_lengths.span_n + 1).cast('int64')
    nbytes = aligned.term.length().cast('float64')
    grouped = joined.group_by(term=aligned.term).agg(
        parent_span_n=aligned.chunk_n.max(),
        nbytes=aligned.term.length().cast('int64').max(),
        estimated_windows=starts.sum().cast('int64'),
        estimated_weighted_bytes=(
            starts.cast('float64')
            * nbytes
            * span_lengths.span_n.cast('float64')
            / aligned.chunk_n.cast('float64')
        )
        .sum()
        .ceil()
        .cast('int64'),
    )
    return grouped.mutate(
        slot=parent_bucket(grouped.term, buckets).cast('int64'),
        estimated_windows=grouped.estimated_windows.cast('int64'),
        estimated_weighted_bytes=grouped.estimated_weighted_bytes.cast('int64'),
    )


def term_stats(patents: Table) -> Table:
    """Return unique-term frequency, patent DF, and JATE word count."""
    return (
        patents
        .unnest('terms')
        .unpack('terms')
        .group_by('term')
        .agg(
            tf=ibis._.frequency.cast('float64').sum().cast('int64'),
            df=ibis._.count().cast('int64'),
        )
        .mutate(word_count=jate_word_count(ibis._.term).cast('int64'))
        .select('term', 'tf', 'df', 'word_count')
    )


def surfaces(patents: Table) -> Table:
    """Return distinct term and normalized surface keys from extract rows."""
    return (
        patents
        .unnest('terms')
        .unpack('terms')
        .unnest('surfaces')
        .mutate(key=normalized_key(ibis._.surfaces))
        .filter(ibis._.key.length() > 0)
        .select('term', 'key')
        .distinct()
    )


def parent_contributions(
    stats: Table,
    span_lengths: Table | None = None,
    *,
    buckets: int = 1,
    bucket: int = 0,
    parents: Table | None = None,
) -> Table:
    """Return unique-parent counts and TF sums keyed by candidate child term.

    Proper windows are distinct per parent. Each parent contributes once to
    a child even when that child substring occurs more than once. Only
    parents with a strictly greater JATE word count are kept. Contributions
    aggregate by window before the candidate join. ``buckets`` and ``bucket``
    select one hash-range parent subset when ``parents`` is omitted; bucket
    count 1 keeps every parent.
    """
    lengths = span_lengths if span_lengths is not None else candidate_span_lengths(stats)
    source = (
        parents
        if parents is not None
        else stats.filter((stats.word_count >= 2) & (parent_bucket(stats.term, buckets) == bucket))
    )
    windows = parent_windows(source, lengths)
    matched = windows.inner_join(stats, windows.window == stats.term)
    bounded = matched.filter(
        (matched.parent_word_count > matched.word_count)
        & jate_contains(matched.parent, matched.window)
    )
    contrib = bounded.group_by(child=bounded.term).agg(
        p_ta=ibis._.count().cast('int64'),
        sum_parent_tf=bounded.parent_tf.sum().cast('float64'),
    )
    return contrib.select(
        child=contrib.child,
        p_ta=contrib.p_ta,
        sum_parent_tf=contrib.sum_parent_tf,
    )


def candidate_parent_contributions(
    stats: Table,
    *,
    parents: Table,
    span_n_min: int,
    span_n_max: int,
) -> Table:
    """Return compact parent contributions by exact candidate containment.

    Candidates stay inside the span band. A parent contributes once per
    child. Containment uses the RE2-safe JATE word-boundary pattern on the
    original strings.
    """
    aligned = chunk_aligned(stats)
    children = aligned.filter(
        (aligned.chunk_n >= span_n_min) & (aligned.chunk_n <= span_n_max)
    ).select(
        child=aligned.term,
        child_word_count=aligned.word_count,
    )
    parent_rows = parents.select(
        parent=parents.term,
        parent_tf=parents.tf,
        parent_word_count=parents.word_count,
    )
    matched = parent_rows.inner_join(
        children,
        (parent_rows.parent_word_count > children.child_word_count)
        & jate_contains(parent_rows.parent, children.child),
    )
    hits = matched.select(
        child=matched.child,
        parent=matched.parent,
        parent_tf=matched.parent_tf,
    ).distinct()
    return hits.group_by(child=hits.child).agg(
        p_ta=ibis._.count().cast('int64'),
        sum_parent_tf=hits.parent_tf.sum().cast('float64'),
    )


def indexed_interval_pairs(
    identity: Table,
    intervals: Table,
    sa_color: Table,
    *,
    rank_lo: int,
    rank_hi: int,
) -> Table:
    """Join exact key intervals to unique SA colors inside one rank band.

    ``sa_color`` is restricted with constant bounds before the range join so
    the parquet scan can skip row groups outside ``[rank_lo, rank_hi)``.
    """
    band = sa_color.filter((sa_color.rank >= rank_lo) & (sa_color.rank < rank_hi))
    hits = (
        intervals
        .join(
            band,
            [band.rank >= intervals.left, band.rank < intervals.right],
        )
        .filter((ibis._.parent_color >= 0) & (ibis._.parent_color != ibis._.color))
        .select('color', 'parent_color')
        .distinct()
    )
    children = identity.select(color=identity.color, child=identity.term)
    parents = identity.select(parent_color=identity.color, parent=identity.term)
    return (
        hits
        .inner_join(children, 'color')
        .inner_join(parents, 'parent_color')
        .select('child', 'parent')
    )


def verified_parent_pairs(stats: Table, pairs: Table) -> Table:
    """Keep original-term pairs that pass word-count and JATE containment."""
    children = stats.select(child=stats.term, child_word_count=stats.word_count)
    parents = stats.select(
        parent=stats.term,
        parent_word_count=stats.word_count,
    )
    joined = pairs.inner_join(children, 'child').inner_join(parents, 'parent')
    matched = joined.filter(
        (joined.parent_word_count > joined.child_word_count)
        & jate_contains(joined.parent, joined.child)
    )
    return matched.select('child', 'parent').distinct()


def compact_parent_pairs(stats: Table, pairs: Table) -> Table:
    """Sum distinct verified parents into compact contribution columns."""
    parent_rows = stats.select(parent=stats.term, parent_tf=stats.tf)
    hits = pairs.inner_join(parent_rows, 'parent').select('child', 'parent', 'parent_tf').distinct()
    return hits.group_by(child=hits.child).agg(
        p_ta=ibis._.count().cast('int64'),
        sum_parent_tf=hits.parent_tf.sum().cast('float64'),
    )


def scored_terms(stats: Table, contrib: Table) -> Table:
    """Return C-value and DF of candidates with at least two JATE words."""
    return (
        stats
        .left_join(contrib, stats.term == contrib.child)
        .filter(ibis._.word_count >= 2)
        .select(
            'term',
            df=ibis._.df.cast('int64'),
            c_value=(
                (ibis._.word_count + 0.1).log2()
                * (ibis._.tf - ibis.coalesce(ibis._.sum_parent_tf / ibis._.p_ta, 0))
            ).cast('float64'),
        )
    )


def cvalue_keys(scored: Table, surface_rows: Table) -> Table:
    """Return max C-value and DF by normalized term and surface key."""
    term_keys = (
        scored
        .mutate(key=normalized_key(ibis._.term))
        .filter(ibis._.key.length() > 0)
        .select('key', 'c_value', 'df')
    )
    surface_scored = surface_rows.inner_join(scored, 'term').select('key', 'c_value', 'df')
    return (
        term_keys
        .union(surface_scored, distinct=False)
        .group_by('key')
        .agg(c_value=ibis._.c_value.max(), df=ibis._.df.max())
    )


def cvalue_expression(patents: Table) -> Table:
    """Compose staged C-value expressions without materializing intermediates."""
    stats = term_stats(patents)
    lengths = candidate_span_lengths(stats)
    return cvalue_keys(
        scored_terms(stats, parent_contributions(stats, lengths)),
        surfaces(patents),
    )


def termhood_table(patents: Table) -> TermhoodTable:
    """Materialize a scored C-value expression into the termhood table."""
    frame = cvalue_expression(patents).to_polars()
    docs = int(patents.n_docs.sum().to_pyarrow().as_py() or 0)
    _log.info('patent_ate.score.cvalue', n_terms=frame.height, n_docs=docs)
    if frame.is_empty():
        return TermhoodTable(total_docs=docs)
    keys = frame['key'].to_list()
    return TermhoodTable(
        c_values=dict(zip(keys, map(float, frame['c_value'].to_list()), strict=True)),
        document_frequency=dict(zip(keys, map(int, frame['df'].to_list()), strict=True)),
        total_docs=docs,
    )

"""Unicode JATE word-class predicates used by window and indexed scoring.

Chunking and parent tests share one RE2-safe word class. A different class
changes C-value.
"""

from __future__ import annotations

import ibis
from ibis.expr.types import BooleanValue, IntegerValue, StringValue

JATE_WORD_CLASS = r'[\p{L}\p{N}_]'
JATE_NONWORD_CLASS = r'[^\p{L}\p{N}_]'
JATE_CHUNK_PATTERN = rf'{JATE_WORD_CLASS}+|{JATE_NONWORD_CLASS}'
JATE_BOUNDARY_PREFIX = rf'(?:^|{JATE_NONWORD_CLASS})'
JATE_BOUNDARY_SUFFIX = rf'(?:$|{JATE_NONWORD_CLASS})'
_SURFACE_EDGE = '.,;:!?()[]{}"\'`-_/\\'


@ibis.udf.scalar.builtin  # type: ignore[untyped-decorator]
def regexp_extract_all(string: str, pattern: str) -> list[str]:
    """DuckDB regexp_extract_all of Unicode word runs and single nonwords."""
    raise NotImplementedError


@ibis.udf.scalar.builtin  # type: ignore[untyped-decorator]
def regexp_escape(string: str) -> str:
    """DuckDB regexp_escape of a literal candidate string."""
    raise NotImplementedError


@ibis.udf.scalar.builtin  # type: ignore[untyped-decorator]
def regexp_matches(string: str, pattern: str, options: str) -> bool:
    """DuckDB regexp_matches with an explicit option string."""
    raise NotImplementedError


def jate_contains(parent: StringValue, child: StringValue) -> BooleanValue:
    r"""Return whether ``parent`` contains ``child`` under JATE word boundaries.

    ``contains`` is only a prefilter. The boundary pattern is the RE2-safe
    form of Python ``(?<!\w)re.escape(child)(?!\w)`` when ``\w`` is
    Unicode alphanumeric plus underscore.
    """
    pattern = (
        ibis.literal(JATE_BOUNDARY_PREFIX)
        + regexp_escape(child)
        + ibis.literal(JATE_BOUNDARY_SUFFIX)
    )
    return parent.contains(child) & regexp_matches(parent, pattern, 'c')


def surface_key(text: str) -> str:
    """Return the Python form of ``normalized_key`` of one original term."""
    return (
        text
        .replace('Ġ', '')
        .replace('▁', '')
        .replace('##', '')
        .strip()
        .lower()
        .strip(_SURFACE_EDGE)
    )


def normalized_key(text: StringValue) -> StringValue:
    """Lowercase lemma key with BPE markers and edge punctuation removed."""

    @ibis.udf.scalar.builtin(name='trim')  # type: ignore[untyped-decorator]
    def trim_chars(string: str, characters: str) -> str:
        raise NotImplementedError

    return trim_chars(
        text.replace('Ġ', '').replace('▁', '').replace('##', '').strip().lower(),
        _SURFACE_EDGE,
    )


def jate_word_count(text: StringValue) -> IntegerValue:
    """Return the whitespace-collapsed token count used by JATE C-value."""
    return text.strip().re_split(r'\s+').length()

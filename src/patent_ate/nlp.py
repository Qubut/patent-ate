"""CPU spaCy host and JATE candidate draws."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from functools import cache, reduce
from itertools import batched, chain, dropwhile, groupby
from operator import attrgetter, itemgetter
from typing import Self

import polars as pl
import spacy
from jate import CValue, MemoryCorpusStore, NounPhraseExtractor, PosPatternExtractor
from jate.features import TermFrequency
from jate.models import Candidate, Document, Term
from pydantic import BaseModel, ConfigDict, Field
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from patent_ate.spec import AteSpec
from patent_ate.termhood import (
    TermhoodTable,
    is_stop_surface,
    normalize_surface,
)

_SEQUENCE_IDENTIFIER = re.compile(r'[:#]\s*\d|\b(?:no|nr|id|seq)\b\.?\s*\d')
_ARTICLES = frozenset({'a', 'an'})
_QUANTITY_NOUNS = frozenset({
    'plurality',
    'number',
    'set',
    'variety',
    'multitude',
    'majority',
    'couple',
})


@cache
def host_language(model: str) -> Language:
    """Return the CPU spaCy Language for ``model``, with NER disabled."""
    return spacy.load(model, disable=['ner'])


class TermSpan(BaseModel):
    """One term interval on the host character grid."""

    model_config = ConfigDict(frozen=True)

    lemma_key: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    token_length: int = Field(ge=1)


class LiveDoc(BaseModel):
    """One non-empty text row with a corpus-global JATE document id."""

    model_config = ConfigDict(frozen=True)

    row: int = Field(ge=0)
    doc_id: str
    text: str


class TextWindow(BaseModel):
    """One spaCy pipe window. Document ids are ``origin + row``."""

    model_config = ConfigDict(frozen=True)

    origin: int = Field(ge=0)
    texts: tuple[str, ...]

    def live(self) -> tuple[LiveDoc, ...]:
        """Return non-empty rows in this window."""
        return tuple(
            LiveDoc(row=index, doc_id=str(self.origin + index), text=text)
            for index, text in enumerate(self.texts)
            if text
        )

    def drawn(self, host: Language, *, keep_spans: bool) -> WindowDraw:
        """Extract JATE candidates for this window."""
        lives = self.live()
        vacant = WindowDraw(candidates={}, docs=tuple(() for _ in self.texts), n_docs=0)
        if not lives:
            return vacant
        documents = tuple(Document(doc_id=doc.doc_id, content=doc.text) for doc in lives)
        parsed = tuple(host.pipe(doc.text for doc in lives))
        candidates = tuple(JateDraw.extract(documents, parsed))
        return WindowDraw(
            candidates=JateDraw.absorb({}, candidates),
            docs=(
                JateDraw.spans(self.texts, candidates, lives, parsed) if keep_spans else vacant.docs
            ),
            n_docs=len(documents),
        )


class WindowDraw(BaseModel):
    """Candidates and term spans from one or more text windows."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    candidates: dict[str, Candidate]
    docs: tuple[tuple[TermSpan, ...], ...]
    n_docs: int = Field(ge=0)

    def merged(self, other: WindowDraw) -> WindowDraw:
        """Fold ``other`` into this draw with JATE ``Candidate.add_position``."""
        return WindowDraw(
            candidates=JateDraw.absorb(self.candidates, tuple(other.candidates.values())),
            docs=self.docs + other.docs,
            n_docs=self.n_docs + other.n_docs,
        )


class JateDraw(BaseModel):
    """Scored term spans for one text batch."""

    model_config = ConfigDict(frozen=True)

    termhood: TermhoodTable
    docs: tuple[tuple[TermSpan, ...], ...]

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        nlp: Language | None = None,
        *,
        min_frequency: int = 1,
        chunk_size: int = 256,
        keep_spans: bool = True,
    ) -> Self:
        """Extract and score terms in ``texts``."""
        empty = cls(termhood=TermhoodTable(), docs=tuple(() for _ in texts))
        if not any(texts):
            return empty
        host = nlp or host_language(AteSpec().spacy_model)
        width = max(1, chunk_size)
        draw = reduce(
            WindowDraw.merged,
            (
                TextWindow(origin=origin, texts=tuple(block)).drawn(host, keep_spans=keep_spans)
                for origin, block in zip(
                    range(0, len(texts), width),
                    batched(texts, width),
                    strict=True,
                )
            ),
        )
        if not draw.candidates:
            return empty
        return cls(
            termhood=cls.score(
                tuple(draw.candidates.values()),
                draw.n_docs,
                min_frequency=min_frequency,
            ),
            docs=draw.docs if keep_spans else empty.docs,
        )

    @staticmethod
    def absorb(
        into: dict[str, Candidate],
        incoming: Sequence[Candidate],
    ) -> dict[str, Candidate]:
        """Fold incoming candidates into ``into`` with JATE ``Candidate.add_position``."""

        def place(
            acc: dict[str, Candidate],
            event: tuple[str, str, frozenset[str], str, int, int, int],
        ) -> dict[str, Candidate]:
            key, surface, surfaces, doc_id, start, end, sent = event
            dest = acc.setdefault(
                key,
                Candidate(surface_form=surface, normalized_form=key),
            )
            dest.surface_forms.update(surfaces)
            dest.add_position(doc_id, start, end, sent)
            return acc

        return reduce(
            place,
            (
                (
                    cand.normalized_form,
                    cand.surface_form,
                    frozenset(cand.surface_forms),
                    doc_id,
                    start,
                    end,
                    sent,
                )
                for cand in incoming
                for doc_id, positions in cand.doc_positions.items()
                for start, end, sent in positions
            ),
            into,
        )

    @staticmethod
    def extract(documents: Sequence[Document], parsed: Sequence[Doc]) -> list[Candidate]:
        """Return unique term candidates for the parsed documents."""

        class ReusedDocs:
            def __init__(self, docs: Mapping[str, Doc]) -> None:
                self._docs = docs

            def process_batch(self, texts: Sequence[str], batch_size: int = 256) -> list[Doc]:
                del batch_size
                return [self._docs[text] for text in texts]

        backend = ReusedDocs({
            document.content: doc for document, doc in zip(documents, parsed, strict=True)
        })
        hosts = {document.doc_id: doc for document, doc in zip(documents, parsed, strict=True)}
        raw = tuple(
            dict(
                chain(
                    (
                        (cand.normalized_form, cand)
                        for cand in NounPhraseExtractor().extract(
                            list(documents), backend, MemoryCorpusStore()
                        )
                    ),
                    (
                        (cand.normalized_form, cand)
                        for cand in PosPatternExtractor().extract(
                            list(documents), backend, MemoryCorpusStore()
                        )
                    ),
                )
            ).values()
        )
        return [
            cand
            for cand in JateDraw.expanded(raw, hosts)
            if len(cand.normalized_form.split()) >= 2
            and not is_stop_surface(normalize_surface(cand.normalized_form))
        ]

    @staticmethod
    def expanded(
        candidates: Sequence[Candidate],
        hosts: Mapping[str, Doc],
    ) -> tuple[Candidate, ...]:
        """Rebuild candidates from admitted pieces of their positions.

        Each position is replaced by the pieces ``pieces`` returns for it, so
        coordinated modifiers gain one candidate per conjunct and fragments
        lose their positions.
        """
        events = dict.fromkeys(
            (key, surface, doc_id, piece_start, piece_end, sent)
            for cand in candidates
            for doc_id, positions in cand.doc_positions.items()
            for start, end, sent in positions
            for key, surface, piece_start, piece_end in JateDraw.pieces(
                hosts[doc_id], start, end, cand.normalized_form, cand.surface_form
            )
        )

        def single(event: tuple[str, str, str, int, int, int]) -> Candidate:
            key, surface, doc_id, start, end, sent = event
            cand = Candidate(surface_form=surface, normalized_form=key)
            cand.add_position(doc_id, start, end, sent)
            return cand

        return tuple(JateDraw.absorb({}, tuple(single(event) for event in events)).values())

    @staticmethod
    def pieces(
        host: Doc,
        start: int,
        end: int,
        key: str,
        surface: str,
    ) -> tuple[tuple[str, str, int, int], ...]:
        """Return the admitted (key, surface, start, end) pieces of one candidate position.

        A position is dropped when its key is a sequence identifier or its span
        opens with a numeric quantifier phrase. A coordinated modifier span is
        replaced by one piece per conjunct; any other position is kept whole.
        """
        span = host.char_span(start, end, alignment_mode='expand')
        if _SEQUENCE_IDENTIFIER.search(key) or (span is not None and JateDraw.quantifier_led(span)):
            return ()
        conjuncts = JateDraw.conjunct_pieces(span) if span is not None else ()
        return conjuncts or ((key, surface, start, end),)

    @staticmethod
    def quantifier_led(span: Span) -> bool:
        """Return whether a span opens with a quantifier phrase rather than a term.

        Covers an adverbial or prepositional opener with a number ('at least one',
        'in one'), a number followed by a coordinator ('one or more'), and a
        quantity noun followed by 'of' after an optional article ('a plurality of',
        'a number of', 'a set of').
        """
        numeric = any(token.like_num or token.pos_ == 'NUM' for token in span) and (
            span[0].pos_ in {'ADP', 'ADV'}
            or (span[0].like_num and any(token.pos_ == 'CCONJ' for token in span[1:2]))
        )
        head = tuple(dropwhile(lambda token: token.lower_ in _ARTICLES, span))
        quantity = (
            len(head) >= 2 and head[0].lemma_.lower() in _QUANTITY_NOUNS and head[1].lower_ == 'of'
        )
        return numeric or quantity

    @staticmethod
    def coordinated_members(span: Span) -> tuple[Token, ...]:
        """Return the conjuncts of the first modifier coordination in ``span``, in order.

        A modifier is a direct dependent of the span root that heads a coordination
        inside the span, and the span must contain a coordinator. Coordination at
        the head itself yields nothing.
        """
        root = span.root
        inside = frozenset(range(span.start, span.end))
        conjunct_sets = tuple(
            members
            for top in root.children
            if top.i in inside
            if len(
                members := tuple(
                    sorted(
                        {token for token in (top, *top.conjuncts) if token.i in inside},
                        key=attrgetter('i'),
                    )
                )
            )
            >= 2
        )
        has_cc = any(token.dep_ == 'cc' for token in span)
        return conjunct_sets[0] if conjunct_sets and has_cc else ()

    @staticmethod
    def conjunct_pieces(span: Span) -> tuple[tuple[str, str, int, int], ...]:
        """Split a coordinated modifier span into one piece per conjunct.

        The shared head is the span root. Each non-last conjunct yields its own
        modifier tokens, keyed with the head lemma; the character offsets cover
        only those tokens because the head is not contiguous with them. The last
        conjunct yields its modifier tokens and the head, which is the contiguous
        tail. Spans with numeric tokens, head coordination or fewer than two
        conjuncts yield nothing, so the caller keeps the span whole.
        """
        if any(token.like_num or token.pos_ == 'NUM' for token in span):
            return ()
        members = JateDraw.coordinated_members(span)
        root = span.root
        inside = frozenset(range(span.start, span.end))
        leading = tuple(
            (core, core)
            for member in members[:-1]
            if (core := JateDraw.modifier_tokens(member, members, inside))
        )
        tail = tuple(
            (core, (*core, root))
            for member in members[-1:]
            if (core := JateDraw.modifier_tokens(member, members, inside))
        )
        return tuple(
            JateDraw.conjunct_piece(core, covered, root) for core, covered in (*leading, *tail)
        )

    @staticmethod
    def modifier_tokens(
        member: Token,
        members: Sequence[Token],
        inside: frozenset[int],
    ) -> tuple[Token, ...]:
        """Return the content tokens of one conjunct, without other conjuncts or coordinators."""
        below = frozenset(token.i for token in member.subtree)
        others = frozenset(
            token.i
            for other in members
            if other is not member and other.i in below
            for token in other.subtree
        )
        return JateDraw.trim_edges(
            tuple(
                token
                for token in member.subtree
                if token.i in inside and token.i not in others and token.dep_ != 'cc'
            )
        )

    @staticmethod
    def trim_edges(tokens: tuple[Token, ...]) -> tuple[Token, ...]:
        """Drop leading and trailing stopwords and punctuation."""

        def is_edge(token: Token) -> bool:
            return token.is_stop or token.is_punct

        head = tuple(dropwhile(is_edge, tokens))
        return tuple(reversed(tuple(dropwhile(is_edge, reversed(head)))))

    @staticmethod
    def conjunct_piece(
        core: tuple[Token, ...],
        covered: tuple[Token, ...],
        root: Token,
    ) -> tuple[str, str, int, int]:
        """Return (key, surface, start, end) for one conjunct over the ``covered`` tokens."""
        start = covered[0].idx
        stop = covered[-1].idx + len(covered[-1].text)
        text = core[0].doc.text
        key = (
            f'{text[core[0].idx : core[-1].idx + len(core[-1].text)].lower()} {root.lemma_.lower()}'
        )
        return key, text[start:stop], start, stop

    @staticmethod
    def score(
        candidates: Sequence[Candidate],
        document_count: int,
        *,
        min_frequency: int,
    ) -> TermhoodTable:
        """Return saturated C-value times Lucene IDF via JATE C-value and Polars."""
        pool = list(candidates)
        freq = TermFrequency.build(pool, document_count)
        ranked = (
            CValue()
            .score(pool, freq)
            .filter_by_frequency(min_frequency)
            .filter_by_length(min_words=2)
        )
        return JateDraw.table_from_cvalue(ranked, freq)

    @staticmethod
    def table_from_cvalue(ranked: Iterable[Term], freq: TermFrequency) -> TermhoodTable:
        """Group JATE C-value terms by surface; scores are a computed field."""
        rows = tuple(
            (key, float(term.score), freq.get_df(term.string))
            for term in ranked
            for surface in (term.string, *term.surface_forms)
            if (key := normalize_surface(surface))
        )
        if not rows:
            return TermhoodTable(total_docs=freq.total_docs)
        frame = (
            pl
            .DataFrame(
                rows,
                schema=['key', 'c_value', 'df'],
                orient='row',
            )
            .group_by('key')
            .agg(pl.col('c_value').max(), pl.col('df').max())
        )
        return TermhoodTable(
            c_values=dict(frame.select('key', 'c_value').iter_rows()),
            document_frequency=dict(frame.select('key', 'df').iter_rows()),
            total_docs=freq.total_docs,
        )

    @staticmethod
    def spans(
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        lives: Sequence[LiveDoc],
        parsed: Sequence[Doc],
    ) -> tuple[tuple[TermSpan, ...], ...]:
        """Align candidate character offsets to host rows."""
        hosts = dict(zip((doc.doc_id for doc in lives), parsed, strict=True))
        rows = {doc.doc_id: doc.row for doc in lives}

        def token_length(doc: Doc, start: int, end: int, key: str) -> int:
            piece = doc.char_span(start, end, alignment_mode='expand')
            return len(piece) if piece is not None else max(1, len(key.split()))

        events = (
            (
                rows[doc_id],
                start,
                TermSpan(
                    lemma_key=key,
                    start_char=start,
                    end_char=end,
                    token_length=width,
                ),
            )
            for cand in candidates
            if (key := normalize_surface(cand.normalized_form)) and not is_stop_surface(key)
            for doc_id, positions in cand.doc_positions.items()
            for start, end, _sent in positions
            if end > start and (width := token_length(hosts[doc_id], start, end, key)) >= 1
        )
        grouped = {
            row: tuple(span for _row, _start, span in group)
            for row, group in groupby(
                sorted(events, key=itemgetter(0, 1)),
                key=itemgetter(0),
            )
        }
        return tuple(grouped.get(index, ()) for index in range(len(texts)))

"""CPU spaCy host and JATE candidate draws."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from functools import cache, reduce
from itertools import batched, chain, compress, dropwhile, groupby, repeat
from operator import attrgetter, itemgetter
from typing import Self

import polars as pl
import spacy
from jate import CValue, MemoryCorpusStore, NounPhraseExtractor, PosPatternExtractor
from jate.extractors.pos_pattern import STOPWORDS
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
_FRAGMENT_KEY = re.compile(
    r'^(?:at\s+)?(?:least|most)\b|^\S+ or more\b'
    r'|^(?:an?\s+)?(?:plurality|number|set|variety|multitude|majority|couple) of\b'
)
_COORDINATORS = frozenset({'and', 'or', 'and/or', '&', '/'})


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
        lose their positions. A position kept whole carries every surface form
        of its candidate; a conjunct piece carries only its own surface.
        """
        events = dict.fromkeys(
            (
                key,
                surface,
                frozenset({
                    surface,
                    *compress(
                        cand.surface_forms,
                        repeat((key, piece_start, piece_end) == (cand.normalized_form, start, end)),
                    ),
                }),
                doc_id,
                piece_start,
                piece_end,
                sent,
            )
            for cand in candidates
            for doc_id, positions in cand.doc_positions.items()
            for start, end, sent in positions
            for key, surface, piece_start, piece_end in JateDraw.pieces(
                hosts[doc_id], start, end, cand.normalized_form, cand.surface_form
            )
        )

        def single(event: tuple[str, str, frozenset[str], str, int, int, int]) -> Candidate:
            key, surface, surfaces, doc_id, start, end, sent = event
            cand = Candidate(surface_form=surface, normalized_form=key)
            cand.surface_forms.update(surfaces)
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

        A position is dropped when its key is a sequence identifier or itself opens
        with a quantifier phrase; a quantifier outside the key leaves the key intact.
        A coordinated modifier span is replaced by one piece per conjunct; any other
        position is kept whole.
        """
        if _SEQUENCE_IDENTIFIER.search(key) or _FRAGMENT_KEY.search(key):
            return ()
        span = host.char_span(start, end, alignment_mode='expand')
        conjuncts = JateDraw.conjunct_pieces(span) if span is not None else ()
        return conjuncts or ((key, surface, start, end),)

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

        The shared head is the span root, and the tokens between the last conjunct
        and the head are shared by every piece. Each non-last conjunct yields its
        own modifier tokens, the shared tokens and the head, keyed with the head
        lemma; its character offsets cover only the conjunct's own tokens because
        the rest is not contiguous with them. The last conjunct yields the
        contiguous tail up to the head. The span stays whole when it has numeric
        tokens or digits (figure labels such as '2a'), a coordinator other than
        'and' or 'or', a conjunct that trims to nothing or ends inside a hyphenated
        word, head coordination, or fewer than two conjuncts.
        """
        if any(
            token.like_num or token.pos_ == 'NUM' or any(char.isdigit() for char in token.text)
            for token in span
        ):
            return ()
        members = JateDraw.coordinated_members(span)
        root = span.root
        inside = frozenset(range(span.start, span.end))
        cores = tuple(JateDraw.modifier_tokens(member, members, inside) for member in members)
        grouped = frozenset(
            token.i for member in members for token in member.subtree if token.i in inside
        )
        shared = JateDraw.trim_edges(
            tuple(
                token
                for token in span
                if token.i > max(grouped, default=span.end)
                and token.i < root.i
                and token.dep_ != 'cc'
            )
        )
        doc = span.doc
        splits_word = any(
            core[-1].i + 1 < len(doc)
            and doc[core[-1].i + 1].text == '-'
            and not core[-1].whitespace_
            for core in cores
            if core
        )
        coordinators = frozenset(token.lower_ for token in span if token.dep_ == 'cc')
        admitted = (
            bool(members)
            and all(cores)
            and coordinators <= _COORDINATORS
            and not splits_word
            and not any(member.is_punct for member in members)
        )
        if not admitted:
            return ()
        return tuple(
            JateDraw.conjunct_piece(core, shared, root, tail=index == len(cores) - 1)
            for index, core in enumerate(cores)
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
        """Drop leading and trailing candidate stopwords and punctuation."""

        def is_edge(token: Token) -> bool:
            return token.lower_ in STOPWORDS or token.is_punct

        head = tuple(dropwhile(is_edge, tokens))
        return tuple(reversed(tuple(dropwhile(is_edge, reversed(head)))))

    @staticmethod
    def conjunct_piece(
        core: tuple[Token, ...],
        shared: tuple[Token, ...],
        root: Token,
        *,
        tail: bool,
    ) -> tuple[str, str, int, int]:
        """Return (key, surface, start, end) for one conjunct.

        The key joins the conjunct's own tokens, the shared tokens and the head
        lemma. The tail conjunct is contiguous with the head, so its surface is
        the source slice up to the head; any other conjunct spells its surface
        from its own tokens, the shared tokens and the head.
        """
        text = core[0].doc.text

        def slice_of(tokens: tuple[Token, ...]) -> str:
            return ' '.join(text[tokens[0].idx : tokens[-1].idx + len(tokens[-1].text)].split())

        words = tuple(slice_of(tokens) for tokens in (core, shared) if tokens)
        key = ' '.join((*(word.lower() for word in words), root.lemma_.lower()))
        start = core[0].idx
        stop = root.idx + len(root.text) if tail else core[-1].idx + len(core[-1].text)
        surface = text[start:stop] if tail else ' '.join((*words, root.text))
        return key, surface, start, stop

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
            if end > start
            and (width := token_length(hosts[doc_id], start, end, key)) >= 1
            and (width >= 2 or len(key.split()) >= 2)
        )
        grouped = {
            row: tuple(span for _row, _start, span in group)
            for row, group in groupby(
                sorted(events, key=itemgetter(0, 1)),
                key=itemgetter(0),
            )
        }
        return tuple(grouped.get(index, ()) for index in range(len(texts)))

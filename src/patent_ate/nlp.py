"""CPU spaCy host and JATE candidate draws."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import cache, reduce
from itertools import batched, chain, groupby
from operator import itemgetter
from typing import Self

import polars as pl
import spacy
from jate import CValue, MemoryCorpusStore, NounPhraseExtractor, PosPatternExtractor
from jate.features import TermFrequency
from jate.models import Candidate, Document, Term
from pydantic import BaseModel, ConfigDict, Field
from spacy.language import Language
from spacy.tokens import Doc

from patent_ate.spec import AteSpec
from patent_ate.termhood import (
    TermhoodTable,
    is_stop_surface,
    normalize_surface,
)


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
        return [
            cand
            for cand in dict(
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
            if len(cand.normalized_form.split()) >= 2
            and not is_stop_surface(normalize_surface(cand.normalized_form))
        ]

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
            if end > start and (width := token_length(hosts[doc_id], start, end, key)) >= 2
        )
        grouped = {
            row: tuple(span for _row, _start, span in group)
            for row, group in groupby(
                sorted(events, key=itemgetter(0, 1)),
                key=itemgetter(0),
            )
        }
        return tuple(grouped.get(index, ()) for index in range(len(texts)))

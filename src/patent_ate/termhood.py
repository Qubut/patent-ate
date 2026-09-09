"""Immutable generation-addressed termhood fact tables."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from functools import cache
from operator import methodcaller
from pathlib import Path
from typing import Self
from uuid import uuid4

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    model_validator,
)

TERMHOOD_PARQUET_NAME = 'termhood.parquet'
TERMHOOD_META_NAME = 'termhood.meta.json'
TERMHOOD_SCHEMA_VERSION = 2
_TERMHOOD_COLUMNS = {'key': pl.String, 'c_value': pl.Float64, 'df': pl.Int64}
_GENERATION = r'^[0-9a-f]{32}$'


class AteLexicon(BaseModel):
    """Published stop surfaces and determiner lemmas for scoring."""

    model_config = ConfigDict(frozen=True)

    source: str
    surfaces: frozenset[str]
    determiners: frozenset[str]

    @classmethod
    @cache
    def from_package(cls) -> Self:
        """Load the packaged stop catalog."""
        return cls.model_validate_json(
            Path(__file__).with_name('ate_stop_surfaces.json').read_text(encoding='utf-8')
        )


def termhood_data_name(generation: str) -> str:
    """Return the immutable generation filename for a fact table."""
    return f'termhood.{generation}.parquet'


def normalize_surface(text: str) -> str:
    """Return a lowercase lemma key with BPE markers and edge punctuation removed."""
    return (
        text
        .replace('Ġ', '')
        .replace('▁', '')
        .replace('##', '')
        .strip()
        .lower()
        .strip('.,;:!?()[]{}"\'`-_/\\')
    )


def is_stop_surface(text: str) -> bool:
    """Return whether every token of ``text`` is punctuation or a published stop."""

    def is_punctuation(part: str) -> bool:
        return bool(part) and all(unicodedata.category(char).startswith('P') for char in part)

    pieces = tuple(part for raw in text.split() if (part := normalize_surface(raw)))
    return (not pieces) or all(
        (not part) or part in AteLexicon.from_package().surfaces or is_punctuation(part)
        for part in pieces
    )


def product_score_expr(total_docs: int) -> pl.Expr:
    """Return ``log1p(C)`` times Lucene IDF; zero when ``df * df > N`` and ``df < N``."""
    docs = total_docs
    return (
        pl
        .when((pl.col('df') <= 0) | ((pl.col('df') * pl.col('df') > docs) & pl.col('df').lt(docs)))
        .then(0.0)
        .otherwise(
            pl.col('c_value').clip(lower_bound=0).log1p()
            * (1.0 + (docs - pl.col('df') + 0.5) / (pl.col('df') + 0.5)).log()
        )
        .alias('score')
    )


class TermhoodTable(BaseModel):
    """Saturated C-value times Lucene IDF, keyed by normalized surface."""

    model_config = ConfigDict(frozen=True)

    c_values: dict[str, float] = Field(default_factory=dict)
    document_frequency: dict[str, int] = Field(default_factory=dict)
    total_docs: int = Field(default=0, ge=0)

    @model_validator(mode='before')
    @classmethod
    def recover_product_c_values(cls, data: object) -> object:
        """Recover C-values from a ``C * log((N+1)/df)`` dump when raw C is missing."""
        if not isinstance(data, Mapping) or data.get('c_values'):
            return data
        scores = data.get('scores')
        freqs = data.get('document_frequency')
        docs = data.get('total_docs')
        if not isinstance(scores, Mapping) or not isinstance(freqs, Mapping) or not docs:
            return data
        keys = tuple(str(key) for key in scores)
        n_docs = int(docs)
        recovered = pl.DataFrame({
            'key': keys,
            'score': tuple(float(scores[key]) for key in scores),
            'df': tuple(int(freqs.get(key, freqs.get(str(key), 0))) for key in scores),
        }).select(
            pl.col('key'),
            pl
            .when((pl.col('score') <= 0) | (pl.col('df') <= 0) | (pl.col('df') >= n_docs))
            .then(0.0)
            .otherwise(pl.col('score') / ((n_docs + 1) / pl.col('df')).log())
            .alias('c_value'),
        )
        return {
            'c_values': dict(recovered.select('key', 'c_value').iter_rows()),
            'document_frequency': {str(key): int(count) for key, count in freqs.items()},
            'total_docs': n_docs,
        }

    @computed_field
    @property
    def scores(self) -> dict[str, float]:
        """``log1p(C)`` times Lucene IDF; zero when ``df * df > N`` and ``df < N``."""
        keys = tuple(self.c_values)
        if not keys or self.total_docs <= 0:
            return {}
        ranked = pl.DataFrame({
            'key': keys,
            'c_value': tuple(self.c_values[key] for key in keys),
            'df': tuple(self.document_frequency.get(key, 0) for key in keys),
        }).select(pl.col('key'), product_score_expr(self.total_docs))
        return dict(ranked.select('key', 'score').iter_rows())

    @classmethod
    def from_texts(cls, texts: Sequence[str], *, min_frequency: int = 1) -> TermhoodTable:
        """Score multiword terms in ``texts``."""
        from patent_ate.nlp import JateDraw  # ruff: ignore[import-outside-top-level]

        return JateDraw.from_texts(texts, min_frequency=min_frequency).termhood

    def scores_for(self, keys: Sequence[str]) -> dict[str, float]:
        """Return product scores for ``keys`` only."""
        wanted = tuple(dict.fromkeys(keys))
        present = tuple(key for key in wanted if key in self.c_values)
        if not wanted:
            return {}
        if not present or self.total_docs <= 0:
            return dict.fromkeys(wanted, 0.0)
        found = dict(
            pl
            .DataFrame({
                'key': present,
                'c_value': tuple(self.c_values[key] for key in present),
                'df': tuple(self.document_frequency.get(key, 0) for key in present),
            })
            .select(pl.col('key'), product_score_expr(self.total_docs))
            .iter_rows()
        )
        return {key: 0.0 if is_stop_surface(key) else float(found.get(key, 0.0)) for key in wanted}

    def score(self, key: str) -> float:
        """Return the termhood of ``key``, or zero for a published stop."""
        return self.scores_for((key,))[key]

    def merged(self, other: TermhoodTable) -> TermhoodTable:
        """Return a table that keeps the larger C-value for each key."""
        frames = tuple(
            pl.DataFrame({
                'key': tuple(table.c_values),
                'c_value': tuple(table.c_values[key] for key in table.c_values),
                'df': tuple(table.document_frequency.get(key, 0) for key in table.c_values),
            })
            for table in (self, other)
            if table.c_values
        )
        if not frames:
            return TermhoodTable(total_docs=max(self.total_docs, other.total_docs))
        frame = pl.concat(frames).group_by('key').agg(pl.col('c_value').max(), pl.col('df').max())
        return TermhoodTable(
            c_values=dict(frame.select('key', 'c_value').iter_rows()),
            document_frequency=dict(frame.select('key', 'df').iter_rows()),
            total_docs=max(self.total_docs, other.total_docs),
        )

    def dump(self) -> dict[str, object]:
        """Return a JSON-safe snapshot of C-values and document frequencies."""
        return self.model_dump(mode='json', exclude={'scores'})

    @classmethod
    def load(cls, payload: Mapping[str, object]) -> Self:
        """Restore a table. Extra or stale keys are ignored."""
        return cls.model_validate(payload)


class TermhoodMeta(BaseModel):
    """Commit manifest for one immutable termhood generation."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    total_docs: int = Field(ge=0)
    n_keys: int = Field(ge=0)
    generation: str = Field(pattern=_GENERATION)
    data_file: str = Field(pattern=r'^termhood\.[0-9a-f]{32}\.parquet$')

    @model_validator(mode='after')
    def data_file_matches_generation(self) -> TermhoodMeta:
        """Refuse a manifest whose data file is not this generation."""
        if self.data_file != termhood_data_name(self.generation):
            raise ValueError('termhood data file does not match generation')
        return self


class TermhoodIndex(BaseModel):
    """One in-memory ``key`` to product-score map built from a committed store."""

    model_config = ConfigDict(frozen=True)

    scores: dict[str, float] = Field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def from_store(cls, store: TermhoodStore) -> Self:
        """Scan the committed fact table once and derive product scores."""
        frame = (
            pl
            .scan_parquet(store.parquet)
            .with_columns(product_score_expr(store.meta.total_docs))
            .select('key', 'score')
            .collect()
        )
        return cls(scores=dict(frame.iter_rows()), source=store.root)

    def scores_for(self, keys: Sequence[str]) -> dict[str, float]:
        """Return product scores for ``keys`` only."""
        wanted = tuple(dict.fromkeys(keys))
        return {
            key: 0.0 if is_stop_surface(key) else float(self.scores.get(key, 0.0)) for key in wanted
        }

    def score(self, key: str) -> float:
        """Return the termhood of ``key``, or zero for a published stop."""
        return self.scores_for((key,))[key]


class TermhoodStore(BaseModel):
    """Committed Parquet fact table plus the generation the manifest names."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    root: Path
    parquet: Path
    meta: TermhoodMeta

    @classmethod
    def clear_partials(cls, artifact_dir: Path) -> None:
        """Remove incomplete parquet and metadata files."""
        leftover = (
            artifact_dir / f'{TERMHOOD_PARQUET_NAME}.partial',
            artifact_dir / f'{TERMHOOD_PARQUET_NAME}.partial.typed',
            artifact_dir / f'{TERMHOOD_META_NAME}.partial',
            *artifact_dir.glob('termhood.*.parquet.partial'),
            *artifact_dir.glob('termhood.*.parquet.partial.typed'),
        )
        _ = tuple(map(methodcaller('unlink', missing_ok=True), leftover))

    @classmethod
    def generations(cls, artifact_dir: Path) -> tuple[Path, ...]:
        """Return committed generation parquet files under ``artifact_dir``."""
        return tuple(sorted(artifact_dir.glob('termhood.*.parquet')))

    @classmethod
    def inspect_parquet(cls, path: Path, *, recast: bool = False) -> int:
        """Return row count after refusing an invalid fact table."""
        if not path.is_file():
            raise ValueError('termhood parquet is missing')
        schema = dict(pl.scan_parquet(path).collect_schema())
        if set(schema) != set(_TERMHOOD_COLUMNS):
            raise ValueError('termhood parquet schema is invalid')
        if schema != _TERMHOOD_COLUMNS:
            if not recast:
                raise ValueError('termhood parquet schema is invalid')
            typed = path.with_name(f'{path.name}.typed')
            pl.scan_parquet(path).select(
                pl.col('key').cast(pl.String),
                pl.col('c_value').cast(pl.Float64),
                pl.col('df').cast(pl.Int64),
            ).sink_parquet(typed)
            typed.replace(path)
        stats = (
            pl
            .scan_parquet(path)
            .select(
                n_keys=pl.len(),
                n_unique=pl.col('key').n_unique(),
                n_bad=(
                    ~pl.col('c_value').is_finite()
                    | pl.col('df').lt(0)
                    | pl.col('key').is_null()
                    | pl.col('key').str.len_bytes().eq(0)
                ).sum(),
            )
            .collect()
        )
        n_keys = int(stats['n_keys'][0])
        if int(stats['n_unique'][0]) != n_keys:
            raise ValueError('termhood parquet keys are not unique')
        if int(stats['n_bad'][0]) > 0:
            raise ValueError('termhood parquet has nonfinite or negative rows')
        return n_keys

    @classmethod
    def commit(cls, parquet_partial: Path, artifact_dir: Path, *, total_docs: int) -> Self:
        """Publish an immutable generation, then replace the manifest last."""
        n_keys = cls.inspect_parquet(parquet_partial, recast=True)
        generation = uuid4().hex
        parquet = artifact_dir / termhood_data_name(generation)
        meta = TermhoodMeta(
            schema_version=TERMHOOD_SCHEMA_VERSION,
            total_docs=total_docs,
            n_keys=n_keys,
            generation=generation,
            data_file=parquet.name,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        meta_partial = artifact_dir / f'{TERMHOOD_META_NAME}.partial'
        _ = meta_partial.write_text(meta.model_dump_json() + '\n', encoding='utf-8')
        parquet_partial.replace(parquet)
        meta_partial.replace(artifact_dir / TERMHOOD_META_NAME)
        return cls(root=artifact_dir, parquet=parquet, meta=meta)

    @classmethod
    def write_frame(cls, frame: pl.DataFrame, artifact_dir: Path, *, total_docs: int) -> Self:
        """Write a small in-memory fact table through the same commit path."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        cls.clear_partials(artifact_dir)
        typed = frame.select(
            pl.col('key').cast(pl.String),
            pl.col('c_value').cast(pl.Float64),
            pl.col('df').cast(pl.Int64),
        )
        partial = artifact_dir / f'{TERMHOOD_PARQUET_NAME}.partial'
        typed.write_parquet(partial)
        return cls.commit(partial, artifact_dir, total_docs=total_docs)

    @classmethod
    def open(cls, path: Path) -> Self:
        """Resolve the manifest's immutable generation, or refuse an incomplete artifact."""
        sidecar = path if path.name == TERMHOOD_META_NAME else path.parent / TERMHOOD_META_NAME
        sidecar = sidecar if path.is_file() else path / TERMHOOD_META_NAME
        if not sidecar.is_file():
            raise ValueError('termhood metadata is missing')
        try:
            meta = TermhoodMeta.model_validate_json(sidecar.read_text(encoding='utf-8'))
        except ValidationError as exc:
            raise ValueError('termhood metadata is invalid') from exc
        parquet = sidecar.parent / meta.data_file
        opened_data = path.is_file() and path.suffix == '.parquet'
        if not parquet.is_file():
            raise ValueError('termhood parquet is missing')
        mismatched = (
            meta.schema_version != TERMHOOD_SCHEMA_VERSION
            or cls.inspect_parquet(parquet, recast=False) != meta.n_keys
            or (opened_data and path.resolve() != parquet.resolve())
        )
        if mismatched:
            raise ValueError('termhood metadata does not match the fact table')
        return cls(root=sidecar.parent, parquet=parquet, meta=meta)

    def report_ranks(
        self,
        k: int,
    ) -> tuple[tuple[tuple[str, float, int], ...], tuple[tuple[str, float, int], ...], int]:
        """Return top-``k`` positive rows by score and by document frequency."""
        positive = (
            pl
            .scan_parquet(self.parquet)
            .with_columns(product_score_expr(self.meta.total_docs))
            .filter(pl.col('score') > 0)
            .select(pl.col('key').alias('label'), 'score', pl.col('df').alias('n_docs'))
        )
        n_unique = int(positive.select(pl.len()).collect().item())

        def rows(order: list[str]) -> tuple[tuple[str, float, int], ...]:
            frame = positive.sort(order, descending=[True, True, False]).head(k).collect()
            return tuple(
                (str(row['label']), float(row['score']), int(row['n_docs']))
                for row in frame.iter_rows(named=True)
            )

        return rows(['score', 'n_docs', 'label']), rows(['n_docs', 'score', 'label']), n_unique

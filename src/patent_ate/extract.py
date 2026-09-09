"""Extract per-patent candidate rows with Ray Data, then score C-value."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from functools import reduce
from itertools import batched, compress, starmap
from operator import attrgetter
from pathlib import Path
from typing import NamedTuple

import polars as pl
import pyarrow as pa
import ray
import structlog
from jate.models import Candidate, Document
from ray.data import ActorPoolStrategy, DataContext, Dataset
from ray.data.checkpoint import CheckpointConfig
from spacy.language import Language
from spacy.tokens import Doc, Span

from patent_ate.corpus import patent_text_from_path
from patent_ate.cvalue import score_term_parquet
from patent_ate.nlp import JateDraw, host_language
from patent_ate.plan import PATENT_ID_COLUMN, PLAN_FILENAME, ExtractPlan
from patent_ate.ray_local import ensure_local_ray
from patent_ate.spec import AteSpec
from patent_ate.termhood import TermhoodStore, TermhoodTable

_log = structlog.get_logger(__name__)

EXTRACT_DIRNAME = 'extract'
CHECKPOINT_DIRNAME = 'ray_checkpoint'
DUCKDB_TMP_DIRNAME = 'duckdb_tmp'

_TERM_STRUCT = pa.struct([
    ('term', pa.string()),
    ('frequency', pa.int64()),
    ('surfaces', pa.list_(pa.string())),
])
_EXTRACT_SCHEMA = pa.schema([
    (PATENT_ID_COLUMN, pa.string()),
    ('n_docs', pa.int64()),
    ('terms', pa.list_(_TERM_STRUCT)),
])


class PatentTermExtractor:
    """One spaCy Language per Ray Data actor. Each patent becomes one compact row."""

    def __init__(self, spacy_model: str, pipe_docs: int, sentence_group: int) -> None:
        self.nlp: Language = host_language(spacy_model)
        self.pipe_docs = max(1, pipe_docs)
        self.sentence_group = max(1, sentence_group)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Extract nested term statistics and keep ``patent_id``."""
        ids = tuple(map(str, batch.column(PATENT_ID_COLUMN).to_pylist()))
        texts, _apps = patent_rows(tuple(map(Path, ids)))
        clipped = tuple(map(self._clipped, texts))
        parsed = self._parse(clipped)
        return pa.Table.from_pylist(
            list(starmap(self._compact, zip(ids, clipped, parsed, strict=True))),
            schema=_EXTRACT_SCHEMA,
        )

    def _clipped(self, text: str) -> str:
        limit = max(1, self.nlp.max_length)
        if len(text) <= limit:
            return text
        _log.warning('patent_ate.extract.clip', n_chars=len(text), limit=limit)
        return text[:limit]

    def _parse(self, texts: Sequence[str]) -> tuple[Doc | None, ...]:
        mask = tuple(map(bool, texts))
        live_texts = tuple(compress(texts, mask))
        live_indices = tuple(compress(range(len(texts)), mask))
        parsed = tuple(self.nlp.pipe(live_texts, batch_size=self.pipe_docs)) if live_texts else ()
        by_index = dict(zip(live_indices, parsed, strict=True))
        return tuple(map(by_index.get, range(len(texts))))

    def _sentence_groups(self, doc: Doc) -> tuple[Doc, ...]:
        sents = tuple(doc.sents)
        if not sents:
            return (doc,)

        def as_group(block: tuple[Span, ...]) -> Doc:
            first, last = block[0], block[-1]
            return doc[first.start : last.end].as_doc()

        return tuple(map(as_group, batched(sents, self.sentence_group)))

    def _compact(self, patent_id: object, text: str, doc: Doc | None) -> dict[str, object]:
        if not text or doc is None:
            return {PATENT_ID_COLUMN: str(patent_id), 'n_docs': 0, 'terms': []}

        def absorb(
            stats: dict[str, tuple[int, set[str]]],
            cand: Candidate,
        ) -> dict[str, tuple[int, set[str]]]:
            key = cand.normalized_form.lower()
            added = sum(map(len, cand.doc_positions.values()))
            frequency, surfaces = stats.get(key, (0, set()))
            return {
                **stats,
                key: (
                    frequency + added,
                    surfaces | set(cand.surface_forms) | {cand.surface_form, cand.normalized_form},
                ),
            }

        pieces = self._sentence_groups(doc)
        empty: dict[str, tuple[int, set[str]]] = {}
        stats = reduce(
            absorb,
            JateDraw.extract(
                tuple(
                    starmap(
                        lambda index, piece: Document(
                            doc_id=f'{patent_id}:{index}',
                            content=piece.text,
                        ),
                        enumerate(pieces),
                    )
                ),
                pieces,
            ),
            empty,
        )
        return {
            PATENT_ID_COLUMN: str(patent_id),
            'n_docs': 1,
            'terms': list(
                starmap(
                    lambda term, packed: {
                        'term': term,
                        'frequency': packed[0],
                        'surfaces': sorted(packed[1]),
                    },
                    stats.items(),
                )
            ),
        }


def resolved_extract_workers(requested: int) -> int:
    """Return a worker count. ``0`` leaves one CPU so Ray can coordinate."""
    counted = getattr(os, 'process_cpu_count', os.cpu_count)
    cpus = counted() or os.cpu_count() or 1
    if requested > 0:
        return max(1, min(requested, cpus))
    return max(1, cpus - 1)


def patent_rows(paths: Sequence[Path | str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load claim text and application numbers of ``paths``."""
    records = tuple(map(patent_text_from_path, map(Path, paths)))
    return (
        tuple(map(attrgetter('text'), records)),
        tuple(map(attrgetter('application_number'), records)),
    )


def extract_checkpoint_config(checkpoint_dir: Path) -> CheckpointConfig:
    """Return Ray Data checkpointing keyed by the plan patent id."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return CheckpointConfig(
        id_column=PATENT_ID_COLUMN,
        checkpoint_path=str(checkpoint_dir),
        delete_checkpoint_on_success=False,
    )


def configure_extract_context(checkpoint_dir: Path) -> DataContext:
    """Install checkpointing, map retries, actor retries, and the 1800s no-progress timeout."""
    context = DataContext.get_current()
    context.checkpoint_config = extract_checkpoint_config(checkpoint_dir)
    context.retried_map_errors = True
    context.max_map_retries = 3
    context.actor_task_retry_on_errors = True
    context.actor_init_retry_on_errors = True
    context.max_errored_blocks = 0
    context.execution_no_progress_timeout_s = 1800
    return context


class PatentPlanRead(NamedTuple):
    """Remaining plan rows as a Ray Data dataset plus the row and block counts."""

    dataset: Dataset
    n_remaining: int
    n_blocks: int


def read_patent_plan(
    plan_path: Path,
    *,
    ate: AteSpec,
    exclude: frozenset[str] = frozenset(),
) -> PatentPlanRead | None:
    """Load remaining plan rows as ``ate.extract_blocks`` Ray Data partitions.

    Returns None when every plan id is already in ``exclude``. Patent-id
    Parquet stays under Ray Data's 1 MiB min block size, and one file is one
    read task. ``from_arrow`` keeps ``override_num_blocks``.
    """
    source = str(plan_path) if plan_path.is_file() else str(plan_path / '*.parquet')
    frame = pl.read_parquet(source)
    if exclude:
        frame = frame.filter(~pl.col(PATENT_ID_COLUMN).cast(pl.String).is_in(tuple(exclude)))
    if frame.height == 0:
        return None
    n_blocks = ate.extract_blocks(frame.height)
    return PatentPlanRead(
        dataset=ray.data.from_arrow(frame.to_arrow(), override_num_blocks=n_blocks),
        n_remaining=frame.height,
        n_blocks=n_blocks,
    )


def extract_parquet_parts(extract_dir: Path) -> tuple[Path, ...]:
    """Return compact extract Parquet files, excluding termhood outputs."""
    return tuple(
        sorted(
            filter(
                lambda path: not path.name.startswith('termhood'),
                extract_dir.glob('*.parquet'),
            )
        )
    )


def extract_ids(extract_dir: Path) -> frozenset[str]:
    """Return patent ids already committed in the extract sink."""
    files = extract_parquet_parts(extract_dir)
    if not files:
        return frozenset()
    frame = (
        pl.scan_parquet(str(extract_dir / '*.parquet')).select(PATENT_ID_COLUMN).unique().collect()
    )
    return frozenset(map(str, frame[PATENT_ID_COLUMN].to_list()))


def run_patent_extract(
    plan_path: Path,
    extract_dir: Path,
    checkpoint_dir: Path,
    *,
    ate: AteSpec,
    workers: int,
    batch_size: int,
) -> None:
    """Read the plan, extract compact rows, and write the checkpointed Parquet sink."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    configure_extract_context(checkpoint_dir)
    loaded = read_patent_plan(plan_path, ate=ate, exclude=extract_ids(extract_dir))
    if loaded is None:
        return
    extracted = loaded.dataset.map_batches(
        PatentTermExtractor,
        batch_size=max(1, batch_size),
        batch_format='pyarrow',
        compute=ActorPoolStrategy(size=max(1, min(workers, loaded.n_blocks))),
        num_cpus=1,
        num_gpus=0,
        fn_constructor_kwargs={
            'spacy_model': ate.spacy_model,
            'pipe_docs': ate.pipe_docs,
            'sentence_group': ate.sentence_group,
        },
        udf_modifying_row_count=False,
        max_restarts=2,
        max_task_retries=2,
    )
    extracted.write_parquet(str(extract_dir), ray_remote_args={'max_retries': 3})


def extract_corpus(
    paths: Sequence[Path],
    *,
    ate: AteSpec,
    chunk_size: int,
    workers: int,
    artifact_dir: Path | None = None,
) -> Path:
    """Write compact extract parquet for ``paths``. Return the extract directory."""
    root = (
        Path(artifact_dir)
        if artifact_dir is not None
        else Path(tempfile.mkdtemp(prefix='patent-ate-'))
    )
    extract_dir = root / EXTRACT_DIRNAME
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not paths:
        return extract_dir
    plan = ExtractPlan.from_paths(paths)
    done = extract_ids(extract_dir)
    if done == plan.ids:
        return extract_dir
    remaining = len(plan.ids - done)
    n_blocks = ate.extract_blocks(remaining)
    n_workers = min(resolved_extract_workers(workers), max(1, n_blocks))
    plan_path = plan.write_parquet(root / PLAN_FILENAME)
    batch_size = max(1, min(chunk_size, ate.cpu_job_width))
    _log.info(
        'patent_ate.extract.start',
        n_paths=len(paths),
        n_jobs=len(plan.rows),
        chunk_size=batch_size,
        blocks=n_blocks,
        extract_block_rows=ate.extract_block_rows,
        workers=n_workers,
        device='cpu',
        spacy_model=ate.spacy_model,
    )
    owns_ray = ensure_local_ray()
    try:
        run_patent_extract(
            plan_path,
            extract_dir,
            root / CHECKPOINT_DIRNAME,
            ate=ate,
            workers=n_workers,
            batch_size=batch_size,
        )
    finally:
        if owns_ray and ray.is_initialized():
            ray.shutdown()
    return extract_dir


def corpus_termhood(
    paths: Sequence[Path],
    *,
    ate: AteSpec,
    chunk_size: int,
    workers: int,
    artifact_dir: Path | None = None,
) -> Path:
    """Score JATE candidates across ``paths`` into a committed termhood artifact."""
    root = (
        Path(artifact_dir)
        if artifact_dir is not None
        else Path(tempfile.mkdtemp(prefix='patent-ate-'))
    )
    if not paths:
        return Path(write_termhood(TermhoodTable(), root))
    extract_dir = extract_corpus(
        paths,
        ate=ate,
        chunk_size=chunk_size,
        workers=workers,
        artifact_dir=root,
    )
    return Path(
        score_term_parquet(
            extract_dir,
            ate=ate,
            temp_dir=root / DUCKDB_TMP_DIRNAME,
            artifact_dir=root,
        )
    )


def write_termhood(table: TermhoodTable, output_dir: Path) -> Path:
    """Write a small in-memory table as one committed termhood artifact directory."""
    if not table.c_values:
        frame = pl.DataFrame(schema={'key': pl.String, 'c_value': pl.Float64, 'df': pl.Int64})
    else:
        keys = pl.DataFrame({
            'key': list(table.c_values),
            'c_value': list(map(float, table.c_values.values())),
        })
        freqs = pl.DataFrame({
            'key': list(table.document_frequency),
            'df': list(map(int, table.document_frequency.values())),
        })
        frame = keys.join(freqs, on='key', how='left').with_columns(
            pl.col('df').fill_null(0).cast(pl.Int64)
        )
    return Path(TermhoodStore.write_frame(frame, output_dir, total_docs=table.total_docs).root)

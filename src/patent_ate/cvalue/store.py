"""Extract fingerprints, score-manifest schemas, and which Parquet files stay valid.

Reusing a unit file after a keep/wipe mismatch scores from stale parents.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from operator import attrgetter, methodcaller
from pathlib import Path
from typing import NamedTuple, Self

import polars as pl
import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from patent_ate.spec import AteSpec

from .plan import (
    INDEXED_ARTIFACT_VERSION,
    INDEXED_CONTRIB_DIRNAME,
    INDEXED_CONTRIB_MANIFEST,
    INDEXED_PLAN_NAME,
    INDEXED_WORKDIR,
    PARENT_PLAN_NAME,
    PARENT_SPAN_COSTS_NAME,
    PARENT_UNIT_DIRNAME,
    SCORED_CHUNK_DIRNAME,
    IndexedContainmentPlan,
    ParentWorkPlan,
)

_log = structlog.get_logger(__name__)

SCORE_STAGE_DIRNAME = 'score_stages'
SCORE_MANIFEST_NAME = 'manifest.json'
SCORE_MANIFEST_VERSION = 1
SCORE_SEMANTIC_VERSION = '6'
RESUME_UNIT_SEMANTICS = frozenset({SCORE_SEMANTIC_VERSION})
CANDIDATE_SPAN_LENGTHS_NAME = 'candidate_span_lengths'
CONTAINMENT_STAGE_NAMES = (
    CANDIDATE_SPAN_LENGTHS_NAME,
    PARENT_SPAN_COSTS_NAME,
    'parent_contrib',
    'scored',
)
TERM_STATS_CAST = {
    'term': 'string',
    'tf': 'int64',
    'df': 'int64',
    'word_count': 'int64',
}
SURFACES_CAST = {'term': 'string', 'key': 'string'}
SPAN_LENGTHS_CAST = {'span_n': 'int64'}
COSTS_CAST = {
    'term': 'string',
    'parent_span_n': 'int64',
    'nbytes': 'int64',
    'estimated_windows': 'int64',
    'estimated_weighted_bytes': 'int64',
    'slot': 'int64',
}
CONTRIB_CAST = {
    'child': 'string',
    'p_ta': 'int64',
    'sum_parent_tf': 'float64',
}
SCORED_CAST = {'term': 'string', 'df': 'int64', 'c_value': 'float64'}
KEYS_CAST = {'key': 'string', 'c_value': 'float64', 'df': 'int64'}
STAGE_CASTS = {
    'term_stats': TERM_STATS_CAST,
    'surfaces': SURFACES_CAST,
    CANDIDATE_SPAN_LENGTHS_NAME: SPAN_LENGTHS_CAST,
    PARENT_SPAN_COSTS_NAME: COSTS_CAST,
    'parent_contrib': CONTRIB_CAST,
    'scored': SCORED_CAST,
}
STAGE_SCHEMAS = {
    'term_stats': pl.Schema({
        'term': pl.String,
        'tf': pl.Int64,
        'df': pl.Int64,
        'word_count': pl.Int64,
    }),
    'surfaces': pl.Schema({'term': pl.String, 'key': pl.String}),
    CANDIDATE_SPAN_LENGTHS_NAME: pl.Schema({'span_n': pl.Int64}),
    PARENT_SPAN_COSTS_NAME: pl.Schema({
        'term': pl.String,
        'parent_span_n': pl.Int64,
        'nbytes': pl.Int64,
        'estimated_windows': pl.Int64,
        'estimated_weighted_bytes': pl.Int64,
        'slot': pl.Int64,
    }),
    'parent_contrib': pl.Schema({
        'child': pl.String,
        'p_ta': pl.Int64,
        'sum_parent_tf': pl.Float64,
    }),
    'scored': pl.Schema({
        'term': pl.String,
        'df': pl.Int64,
        'c_value': pl.Float64,
    }),
}


class ExtractFingerprint(BaseModel):
    """Exact identity of the compact extract files a score run read."""

    model_config = ConfigDict(frozen=True)

    n_files: int = Field(ge=0)
    sum_bytes: int = Field(ge=0)
    digest: str = Field(min_length=64, max_length=64, pattern=r'^[0-9a-f]{64}$')

    @classmethod
    def from_files(cls, files: Sequence[Path]) -> Self:
        """Digest sorted names, sizes, and nanosecond mtimes of ``files``."""

        class _FileStamp(NamedTuple):
            name: str
            size: int
            mtime_ns: int

            def token(self) -> bytes:
                return b''.join((
                    self.name.encode(),
                    b'\0',
                    str(self.size).encode(),
                    b'\0',
                    str(self.mtime_ns).encode(),
                    b'\n',
                ))

        def stamp_of(path: Path) -> _FileStamp:
            stat = path.stat()
            return _FileStamp(path.name, stat.st_size, stat.st_mtime_ns)

        stamps = tuple(map(stamp_of, files))
        return cls(
            n_files=len(stamps),
            sum_bytes=sum(map(attrgetter('size'), stamps)),
            digest=hashlib.sha256(b''.join(map(_FileStamp.token, stamps))).hexdigest(),
        )


class IndexedMapsFingerprint(BaseModel):
    """Inode and size identity of the three indexed interval-map files."""

    model_config = ConfigDict(frozen=True)

    digest: str = Field(min_length=64, max_length=64, pattern=r'^[0-9a-f]{64}$')
    n_chunks: int = Field(ge=1)

    @classmethod
    def from_maps(cls, work: Path, n_chunks: int) -> Self:
        """Digest names, inodes, sizes, and nanosecond mtimes of the map files."""

        class _MapStamp(NamedTuple):
            name: str
            ino: int
            size: int
            mtime_ns: int

            def token(self) -> bytes:
                return b''.join((
                    self.name.encode(),
                    b'\0',
                    str(self.ino).encode(),
                    b'\0',
                    str(self.size).encode(),
                    b'\0',
                    str(self.mtime_ns).encode(),
                    b'\n',
                ))

        names = ('key_intervals.parquet', 'sa_color.parquet', 'term_identity.parquet')

        def stamp_of(name: str) -> _MapStamp:
            stat = (work / name).stat()
            return _MapStamp(name, stat.st_ino, stat.st_size, stat.st_mtime_ns)

        stamps = tuple(map(stamp_of, names))
        return cls(
            digest=hashlib.sha256(b''.join(map(_MapStamp.token, stamps))).hexdigest(),
            n_chunks=n_chunks,
        )


class ScoreStageManifest(BaseModel):
    """Completed score-stage files of one extract fingerprint and scorer config."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(default=SCORE_MANIFEST_VERSION, ge=1)
    semantic_version: str = Field(min_length=1)
    extract: ExtractFingerprint
    duckdb_memory: str = Field(min_length=1)
    duckdb_threads: int = Field(ge=1)
    parent_buckets: int = Field(ge=1)
    tail_window_cap: int = Field(default=400_000, ge=1)
    tail_weighted_bytes_cap: int = Field(default=400_000_000, ge=1)
    base_window_cap: int = Field(default=25_000_000, ge=1)
    base_weighted_bytes_cap: int = Field(default=2_500_000_000, ge=1)
    max_parent_units: int = Field(default=64, ge=1)
    max_candidate_scans: int = Field(default=64, ge=1)
    tail_candidate_row_cap: int = Field(default=50_000, ge=1)
    tail_compare_cap: int = Field(default=50_000, ge=1)
    containment: str = 'window'
    indexed_artifact_version: str = ''
    gsa_engine: str = ''
    gsa_prefix_bytes: int = 0
    total_docs: int = Field(ge=0)
    completed: tuple[str, ...] = ()
    completed_units: tuple[str, ...] = ()

    @staticmethod
    def gsa_fields(ate: AteSpec) -> tuple[str, str, int]:
        """Return indexed artifact version, engine, and prefix width for ``ate``."""
        spec = ate.indexed
        if ate.containment == 'indexed' and spec is not None:
            return INDEXED_ARTIFACT_VERSION, spec.engine, spec.prefix_bytes
        return '', '', 0

    @classmethod
    def from_ate(
        cls,
        *,
        extract: ExtractFingerprint,
        ate: AteSpec,
        total_docs: int,
        completed: tuple[str, ...] = (),
        completed_units: tuple[str, ...] = (),
    ) -> Self:
        """Build a manifest from the current scorer identity and progress."""
        indexed_artifact_version, gsa_engine, gsa_prefix_bytes = cls.gsa_fields(ate)
        return cls(
            extract=extract,
            duckdb_memory=ate.duckdb_memory,
            duckdb_threads=ate.duckdb_threads,
            semantic_version=SCORE_SEMANTIC_VERSION,
            parent_buckets=ate.parent_buckets,
            tail_window_cap=ate.tail_window_cap,
            tail_weighted_bytes_cap=ate.tail_weighted_bytes_cap,
            base_window_cap=ate.base_window_cap,
            base_weighted_bytes_cap=ate.base_weighted_bytes_cap,
            max_parent_units=ate.max_parent_units,
            max_candidate_scans=ate.max_candidate_scans,
            tail_candidate_row_cap=ate.tail_candidate_row_cap,
            tail_compare_cap=ate.tail_compare_cap,
            containment=ate.containment,
            indexed_artifact_version=indexed_artifact_version,
            gsa_engine=gsa_engine,
            gsa_prefix_bytes=gsa_prefix_bytes,
            total_docs=total_docs,
            completed=completed,
            completed_units=completed_units,
        )

    def public_prefix_matches(self, extract: ExtractFingerprint, ate: AteSpec) -> bool:
        """Return whether term_stats and surfaces may be reused."""
        return (
            self.schema_version == SCORE_MANIFEST_VERSION
            and self.extract == extract
            and self.duckdb_memory == ate.duckdb_memory
            and self.duckdb_threads == ate.duckdb_threads
        )

    def unit_config_matches(self, ate: AteSpec) -> bool:
        """Return whether parent units and merge may resume under ``ate``."""
        indexed_artifact_version, gsa_engine, gsa_prefix_bytes = self.gsa_fields(ate)
        indexed_ok = (
            self.indexed_artifact_version == indexed_artifact_version
            and self.gsa_engine == gsa_engine
            and self.gsa_prefix_bytes == gsa_prefix_bytes
        )
        return (
            self.semantic_version in RESUME_UNIT_SEMANTICS
            and self.containment == ate.containment
            and indexed_ok
            and self.parent_buckets == ate.parent_buckets
            and self.tail_window_cap == ate.tail_window_cap
            and self.tail_weighted_bytes_cap == ate.tail_weighted_bytes_cap
            and self.base_window_cap == ate.base_window_cap
            and self.base_weighted_bytes_cap == ate.base_weighted_bytes_cap
            and self.max_parent_units == ate.max_parent_units
            and self.max_candidate_scans == ate.max_candidate_scans
            and self.tail_candidate_row_cap == ate.tail_candidate_row_cap
            and self.tail_compare_cap == ate.tail_compare_cap
        )

    def matches(self, extract: ExtractFingerprint, ate: AteSpec) -> bool:
        """Return whether this manifest may resume the current score identity."""
        return self.public_prefix_matches(extract, ate) and self.unit_config_matches(ate)

    def advance(self, ate: AteSpec, *, resume_units: bool) -> Self:
        """Keep only stages that remain valid under ``ate``."""
        keep = {'term_stats', 'surfaces'}
        if resume_units:
            keep |= {
                CANDIDATE_SPAN_LENGTHS_NAME,
                PARENT_SPAN_COSTS_NAME,
                'parent_contrib',
                'scored',
            }
        return self.from_ate(
            extract=self.extract,
            ate=ate,
            total_docs=self.total_docs,
            completed=tuple(filter(keep.__contains__, self.completed)),
            completed_units=self.completed_units if resume_units else (),
        )


class ScoreStageStore:
    """Artifact paths, Parquet validity, and resume wipe policy of one dest."""

    def __init__(self, dest: Path) -> None:
        self.dest = dest
        self.stages_dir: Path = dest / SCORE_STAGE_DIRNAME
        self.manifest_path: Path = self.stages_dir / SCORE_MANIFEST_NAME

    @staticmethod
    def extract_files(extract_dir: Path) -> tuple[Path, ...]:
        """Return compact extract Parquet files, excluding termhood outputs."""
        return tuple(
            sorted(
                filter(
                    lambda path: not path.name.startswith('termhood'),
                    extract_dir.glob('*.parquet'),
                )
            )
        )

    @staticmethod
    def parquet_valid(path: Path, expected: pl.Schema) -> bool:
        """Return whether ``path`` is a non-empty Parquet file of ``expected``."""
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        return pl.scan_parquet(path).collect_schema() == expected

    def stage_parquet(self, name: str) -> Path:
        """Return the Parquet path of a declared score stage."""
        return self.stages_dir / f'{name}.parquet'

    def stage_valid(self, name: str) -> bool:
        """Return whether the named stage Parquet matches the declared schema."""
        return self.parquet_valid(self.stage_parquet(name), STAGE_SCHEMAS[name])

    def unit_path(self, unit_id: str) -> Path:
        """Return the compact contribution Parquet of ``unit_id``."""
        return Path(self.stages_dir, PARENT_UNIT_DIRNAME, f'{unit_id}.parquet')

    def unit_valid(self, unit_id: str) -> bool:
        """Return whether a unit contribution file matches the merge schema."""
        return self.parquet_valid(self.unit_path(unit_id), STAGE_SCHEMAS['parent_contrib'])

    def bind(self, extract_dir: Path, ate: AteSpec) -> ScoreStageSession:
        """Open or invalidate stages of ``extract_dir`` under ``ate``."""
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        files = self.extract_files(extract_dir)
        fingerprint = ExtractFingerprint.from_files(files)
        try:
            loaded = ScoreStageManifest.model_validate_json(
                self.manifest_path.read_text(encoding='utf-8')
            )
        except (OSError, ValidationError, ValueError):
            loaded = None
        prior_buckets = loaded.parent_buckets if loaded is not None else ate.parent_buckets
        if loaded is None or not loaded.public_prefix_matches(fingerprint, ate):
            self.wipe_files(self.stages_dir.rglob('*'))
            loaded = ScoreStageManifest.from_ate(extract=fingerprint, ate=ate, total_docs=0)
            resume_units = False
        else:
            resume_units = loaded.unit_config_matches(ate)
            loaded = loaded.advance(ate, resume_units=resume_units)
            if not resume_units:
                self.wipe_containment()
        self.clear_partials()
        return ScoreStageSession(
            store=self,
            extract_files=files,
            fingerprint=fingerprint,
            ate=ate,
            resume_units=resume_units,
            completed=list(filter(lambda name: name != 'term_stats', loaded.completed)),
            finished_units=list(loaded.completed_units) if resume_units else [],
            total_docs=loaded.total_docs,
            prior_parent_buckets=prior_buckets,
        )

    def attach(
        self,
        ate: AteSpec,
        *,
        extract_files: tuple[Path, ...],
        fingerprint: ExtractFingerprint,
        resume_units: bool,
        completed_units: tuple[str, ...],
        total_docs: int,
    ) -> ScoreStageSession:
        """Rebuild a score session from a published plan and current manifest."""
        loaded = ScoreStageManifest.model_validate_json(
            self.manifest_path.read_text(encoding='utf-8')
        )
        return ScoreStageSession(
            store=self,
            extract_files=extract_files,
            fingerprint=fingerprint,
            ate=ate,
            resume_units=resume_units,
            completed=list(loaded.completed),
            finished_units=list(completed_units),
            total_docs=total_docs,
            prior_parent_buckets=loaded.parent_buckets,
        )

    def publish(self, manifest: ScoreStageManifest) -> None:
        """Replace the stage manifest through a ``.partial`` file."""
        partial = self.stages_dir / f'{SCORE_MANIFEST_NAME}.partial'
        _ = partial.write_text(manifest.model_dump_json() + '\n', encoding='utf-8')
        partial.replace(self.manifest_path)

    def wipe_files(self, paths: Iterable[Path]) -> None:
        """Unlink existing files in ``paths``."""
        tuple(map(methodcaller('unlink', missing_ok=True), filter(Path.is_file, paths)))

    def wipe_containment(self) -> None:
        """Drop units, the work plan, and containment-stage Parquet files."""
        unit_dir = self.stages_dir / PARENT_UNIT_DIRNAME
        indexed_dir = self.stages_dir / INDEXED_WORKDIR
        scored_dir = self.stages_dir / SCORED_CHUNK_DIRNAME
        self.wipe_files(unit_dir.rglob('*') if unit_dir.exists() else ())
        self.wipe_files(indexed_dir.rglob('*') if indexed_dir.exists() else ())
        self.wipe_files(scored_dir.rglob('*') if scored_dir.exists() else ())
        (self.stages_dir / PARENT_PLAN_NAME).unlink(missing_ok=True)
        (self.stages_dir / INDEXED_PLAN_NAME).unlink(missing_ok=True)
        tuple(
            map(
                methodcaller('unlink', missing_ok=True),
                map(self.stage_parquet, CONTAINMENT_STAGE_NAMES),
            )
        )

    def wipe_units_and_scored(self) -> None:
        """Drop unit files and merged contribution/score Parquet."""
        unit_dir = self.stages_dir / PARENT_UNIT_DIRNAME
        chunk_dir = self.stages_dir / INDEXED_WORKDIR / INDEXED_CONTRIB_DIRNAME
        scored_dir = self.stages_dir / SCORED_CHUNK_DIRNAME
        indexed_dir = self.stages_dir / INDEXED_WORKDIR
        self.wipe_files(unit_dir.rglob('*') if unit_dir.exists() else ())
        self.wipe_files(chunk_dir.rglob('*') if chunk_dir.exists() else ())
        self.wipe_files(scored_dir.rglob('*') if scored_dir.exists() else ())
        (indexed_dir / INDEXED_CONTRIB_MANIFEST).unlink(missing_ok=True)
        (indexed_dir / 'interval_chunks.parquet').unlink(missing_ok=True)
        tuple(
            map(
                methodcaller('unlink', missing_ok=True),
                map(self.stage_parquet, ('parent_contrib', 'scored')),
            )
        )

    def clear_partials(self) -> None:
        """Remove leftover ``*.partial`` files under the stage directory."""
        self.wipe_files(self.stages_dir.rglob('*.partial'))

    def load_matching_plan(self, ate: AteSpec) -> ParentWorkPlan | None:
        """Return a stored work plan only when it still matches ``ate``."""
        path = self.stages_dir / PARENT_PLAN_NAME
        if not path.is_file():
            return None
        try:
            loaded = ParentWorkPlan.model_validate_json(path.read_text(encoding='utf-8'))
        except (OSError, ValidationError, ValueError):
            return None
        return loaded if loaded.matches(ate) else None

    def write_filters(
        self,
        filter_terms: dict[str, tuple[str, ...]],
        *,
        resume_units: bool,
    ) -> None:
        """Write exact parent-filter Parquet, reusing identical files."""
        unit_dir = self.stages_dir / PARENT_UNIT_DIRNAME
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir / 'filters').mkdir(parents=True, exist_ok=True)

        def persist_filter(item: tuple[str, tuple[str, ...]]) -> None:
            rel, terms = item
            out = self.stages_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            existing = (
                tuple(map(str, pl.scan_parquet(out).select('term').collect()['term'].to_list()))
                if resume_units and out.is_file()
                else ()
            )
            if existing == terms and out.is_file():
                return
            partial = out.with_name(f'{out.name}.partial')
            partial.unlink(missing_ok=True)
            pl.DataFrame({'term': list(terms)}, schema={'term': pl.String}).write_parquet(partial)
            partial.replace(out)

        tuple(map(persist_filter, filter_terms.items()))

    def replace_plan(self, plan: ParentWorkPlan) -> None:
        """Publish ``plan`` through a ``.partial`` file."""
        path = self.stages_dir / PARENT_PLAN_NAME
        partial = path.with_name(f'{path.name}.partial')
        _ = partial.write_text(plan.model_dump_json() + '\n', encoding='utf-8')
        partial.replace(path)

    def indexed_dir(self) -> Path:
        """Return the staged work directory of indexed candidate generation."""
        dest = Path(self.stages_dir, INDEXED_WORKDIR)
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def load_matching_indexed_plan(self, ate: AteSpec) -> IndexedContainmentPlan | None:
        """Return a stored indexed plan only when it still matches ``ate``."""
        path = self.stages_dir / INDEXED_PLAN_NAME
        if not path.is_file():
            return None
        try:
            loaded = IndexedContainmentPlan.model_validate_json(path.read_text(encoding='utf-8'))
        except (OSError, ValidationError, ValueError):
            return None
        return loaded if loaded.matches(ate, semantic_version=SCORE_SEMANTIC_VERSION) else None

    def replace_indexed_plan(self, plan: IndexedContainmentPlan) -> None:
        """Publish an indexed containment plan through a ``.partial`` file."""
        path = self.stages_dir / INDEXED_PLAN_NAME
        partial = path.with_name(f'{path.name}.partial')
        _ = partial.write_text(plan.model_dump_json() + '\n', encoding='utf-8')
        partial.replace(path)

    def load_indexed_contrib_identity(self, work: Path) -> IndexedMapsFingerprint | None:
        """Return the stored maps fingerprint when the chunk manifest is valid."""
        path = work / INDEXED_CONTRIB_MANIFEST
        if not path.is_file():
            return None
        try:
            return IndexedMapsFingerprint.model_validate_json(path.read_text(encoding='utf-8'))
        except (OSError, ValidationError, ValueError):
            return None

    def replace_indexed_contrib_identity(
        self,
        work: Path,
        identity: IndexedMapsFingerprint,
    ) -> None:
        """Publish the maps fingerprint that owned the current contrib chunks."""
        path = work / INDEXED_CONTRIB_MANIFEST
        partial = path.with_name(f'{path.name}.partial')
        _ = partial.write_text(identity.model_dump_json() + '\n', encoding='utf-8')
        partial.replace(path)

    def bind_indexed_contrib_chunks(self, work: Path, stamp: IndexedMapsFingerprint) -> Path:
        """Return the contrib-chunk directory, wiping parts when maps identity changed."""
        chunk_dir = work / INDEXED_CONTRIB_DIRNAME
        chunk_dir.mkdir(parents=True, exist_ok=True)
        prior = self.load_indexed_contrib_identity(work)
        if prior != stamp:
            self.wipe_files(chunk_dir.rglob('*') if chunk_dir.exists() else ())
            self.replace_indexed_contrib_identity(work, stamp)
        return chunk_dir


@dataclass(slots=True)
class ScoreStageSession:
    """Mutable resume cursor of one bind or attach of a score dest."""

    store: ScoreStageStore
    extract_files: tuple[Path, ...]
    fingerprint: ExtractFingerprint
    ate: AteSpec
    resume_units: bool
    completed: list[str]
    finished_units: list[str]
    total_docs: int
    prior_parent_buckets: int

    def stage_valid(self, name: str) -> bool:
        """Return whether the named stage Parquet is reusable."""
        return self.store.stage_valid(name)

    def finish(self, name: str, units: tuple[str, ...] | None = None) -> None:
        """Record ``name`` and optional unit ids, then publish the manifest."""
        if name not in self.completed:
            self.completed.append(name)
        if units is not None:
            self.finished_units.clear()
            self.finished_units.extend(units)
        self.store.publish(
            ScoreStageManifest.from_ate(
                extract=self.fingerprint,
                ate=self.ate,
                total_docs=self.total_docs,
                completed=tuple(self.completed),
                completed_units=tuple(self.finished_units),
            )
        )

    def note_resume(self, name: str, path: Path | None = None) -> None:
        """Log a reused stage or unit Parquet without rewriting it."""
        dest = path if path is not None else self.store.stage_parquet(name)
        n_rows = int(pl.scan_parquet(dest).select(pl.len()).collect().item())
        _log.info(
            'patent_ate.score.stage',
            stage=name,
            resumed=True,
            wall_s=0.0,
            n_rows=n_rows,
            output_bytes=dest.stat().st_size,
            spill_bytes=0,
        )

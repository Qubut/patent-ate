"""Validated envelopes for extract, score, and corpus termhood runs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from patent_ate.corpus import sample_json_paths
from patent_ate.cvalue import score_term_parquet
from patent_ate.extract import DUCKDB_TMP_DIRNAME, EXTRACT_DIRNAME, corpus_termhood, extract_corpus
from patent_ate.spec import AteSpec


class AteRunRequest(BaseModel):
    """Typed inputs that extract or score C-value on a JSON patent sample."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    output_dir: Path
    config_path: Path | None = None
    input_dir: Path | None = None
    index_cache: Path | None = None
    limit: int = Field(default=1000, ge=1)
    extract_chunk_size: int = Field(default=64, ge=1)
    extract_workers: int = Field(default=0, ge=0)
    extract_block_rows: int | None = Field(default=None, ge=1)
    seed: int = 20260830
    ate: AteSpec | None = None


def resolved_ate(request: AteRunRequest) -> AteSpec:
    """Return the request spec, with optional extract-block override."""
    ate = request.ate if request.ate is not None else AteSpec.from_yaml(request.config_path)
    if request.extract_block_rows is None:
        return ate
    return AteSpec.model_validate({
        **ate.model_dump(),
        'extract_block_rows': request.extract_block_rows,
    })


def run_corpus_ate(request: AteRunRequest) -> Path:
    """Sample JSON patent paths and write the committed termhood artifact."""
    if request.input_dir is None:
        raise ValueError('input_dir is required')
    paths, _n_pool = sample_json_paths(
        request.input_dir,
        limit=request.limit,
        seed=request.seed,
        index_cache=request.index_cache,
    )
    return Path(
        corpus_termhood(
            paths,
            ate=resolved_ate(request),
            chunk_size=request.extract_chunk_size,
            workers=request.extract_workers,
            artifact_dir=request.output_dir,
        )
    )


def run_extract(request: AteRunRequest) -> Path:
    """Sample JSON patent paths and write compact extract Parquet without scoring."""
    if request.input_dir is None:
        raise ValueError('input_dir is required')
    paths, _n_pool = sample_json_paths(
        request.input_dir,
        limit=request.limit,
        seed=request.seed,
        index_cache=request.index_cache,
    )
    return Path(
        extract_corpus(
            paths,
            ate=resolved_ate(request),
            chunk_size=request.extract_chunk_size,
            workers=request.extract_workers,
            artifact_dir=request.output_dir,
        )
    )


class AteScoreRequest(BaseModel):
    """Score compact extract Parquet into a committed termhood artifact."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    extract_dir: Path
    output_dir: Path
    config_path: Path | None = None
    ate: AteSpec | None = None


def run_score(request: AteScoreRequest) -> Path:
    """Score an existing extract directory into ``output_dir``."""
    ate = request.ate if request.ate is not None else AteSpec.from_yaml(request.config_path)
    extract_dir = (
        request.extract_dir
        if request.extract_dir.name == EXTRACT_DIRNAME
        else request.extract_dir / EXTRACT_DIRNAME
        if (request.extract_dir / EXTRACT_DIRNAME).is_dir()
        else request.extract_dir
    )
    return Path(
        score_term_parquet(
            extract_dir,
            ate=ate,
            temp_dir=request.output_dir / DUCKDB_TMP_DIRNAME,
            artifact_dir=request.output_dir,
        )
    )

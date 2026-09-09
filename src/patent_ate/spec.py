"""Extract and C-value scoring settings."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class IndexedContainmentSpec(BaseModel):
    """Generalized suffix-array listing used when containment is indexed."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    engine: Literal['gsa'] = 'gsa'
    prefix_bytes: int = Field(default=8, ge=8, le=8)


class AteSpec(BaseModel):
    """CPU extract and C-value scoring settings."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    spacy_model: str = Field(default='en_core_web_lg', min_length=1)
    cpu_job_width: int = Field(default=8, ge=1)
    extract_block_rows: int = Field(default=256, ge=1)
    pipe_docs: int = Field(default=128, ge=1)
    sentence_group: int = Field(default=32, ge=1)
    duckdb_memory: str = Field(default='8GB', min_length=1)
    duckdb_threads: int = Field(default=8, ge=1)
    parent_buckets: int = Field(default=32, ge=1)
    tail_window_cap: int = Field(default=400_000, ge=1)
    tail_weighted_bytes_cap: int = Field(default=400_000_000, ge=1)
    base_window_cap: int = Field(default=25_000_000, ge=1)
    base_weighted_bytes_cap: int = Field(default=2_500_000_000, ge=1)
    max_parent_units: int = Field(default=64, ge=1)
    max_candidate_scans: int = Field(default=64, ge=1)
    tail_candidate_row_cap: int = Field(default=50_000, ge=1)
    tail_compare_cap: int = Field(default=50_000, ge=1)
    containment: Literal['window', 'indexed'] = 'window'
    indexed: IndexedContainmentSpec | None = None

    @model_validator(mode='after')
    def block_rows_cover_job_width(self) -> Self:
        """Refuse a plan block smaller than the extract batch."""
        if self.extract_block_rows < self.cpu_job_width:
            raise ValueError('extract_block_rows must be at least cpu_job_width')
        if self.containment == 'indexed' and self.indexed is None:
            raise ValueError('indexed containment requires indexed settings')
        return self

    def extract_blocks(self, n_remaining: int) -> int:
        """Return how many extract plan blocks cover ``n_remaining`` patent rows."""
        if n_remaining <= 0:
            return 0
        return min(n_remaining, math.ceil(n_remaining / self.extract_block_rows))

    @classmethod
    def from_yaml(cls, path: Path | None) -> Self:
        """Load from YAML, or return package defaults when ``path`` is omitted."""
        if path is None:
            return cls()
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
        if payload is None:
            return cls()
        return cls.model_validate(payload)

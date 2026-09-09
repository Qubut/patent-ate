"""One-row-per-patent extract plan written as Parquet."""

from __future__ import annotations

from collections.abc import Sequence
from operator import attrgetter
from pathlib import Path
from typing import Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

PATENT_ID_COLUMN = 'patent_id'
PLAN_FILENAME = 'plan.parquet'


class PatentPlanRow(BaseModel):
    """One patent path and the unique id Ray Data checkpointing keeps."""

    model_config = ConfigDict(frozen=True)

    patent_id: str = Field(min_length=1)
    path: str = Field(min_length=1)


class ExtractPlan(BaseModel):
    """Validated patent rows that become the Ray Data input dataset."""

    model_config = ConfigDict(frozen=True)

    rows: tuple[PatentPlanRow, ...] = ()

    @model_validator(mode='after')
    def _unique_patent_ids(self) -> Self:
        ids = tuple(map(attrgetter('patent_id'), self.rows))
        if len(ids) != len(set(ids)):
            msg = 'patent_id values must be unique'
            raise ValueError(msg)
        return self

    @classmethod
    def from_paths(cls, paths: Sequence[Path]) -> Self:
        """Build one row per resolved path, using the path as the stable id."""
        resolved = tuple(map(Path.resolve, map(Path, paths)))

        def as_row(path: Path) -> PatentPlanRow:
            text = str(path)
            return PatentPlanRow(patent_id=text, path=text)

        return cls(rows=tuple(map(as_row, resolved)))

    @property
    def ids(self) -> frozenset[str]:
        """Unique patent ids in this plan."""
        return frozenset(map(attrgetter('patent_id'), self.rows))

    def write_parquet(self, path: Path) -> Path:
        """Write the plan as one Parquet file and return that path.

        The Ray Data parquet reader does not keep a column named ``path``.
        The stable id is the resolved patent file, so extract reads ``patent_id``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            PATENT_ID_COLUMN: tuple(map(attrgetter('patent_id'), self.rows)),
        }).write_parquet(path)
        return path

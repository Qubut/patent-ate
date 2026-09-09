"""JSON patent text: claims, abstract, and summary fields.

One well-known dump that uses this object shape is the Harvard USPTO Patent
Dataset (HUPD): https://huggingface.co/datasets/HUPD/hupd
(Suzgun et al., 2022, https://arxiv.org/abs/2207.04043). Any directory of JSON
objects with those keys is accepted.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def strip_soh_markup(text: str) -> str:
    """Remove ``<SOH>`` / ``<EOH>`` wrappers some USPTO JSON dumps insert."""
    return text.replace('<SOH>', '').replace('<EOH>', '').strip()


class PatentText(BaseModel):
    """Claim, abstract, and summary text of one patent JSON file."""

    model_config = ConfigDict(frozen=True)

    application_number: str = Field(min_length=0)
    text: str


def _field_text(raw: object) -> str:
    if raw is None:
        return ''
    return str(raw).strip()


def patent_text_from_path(path: Path) -> PatentText:
    """Load claim, abstract, and summary text from one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        msg = 'patent JSON must be an object'
        raise TypeError(msg)
    claims = _field_text(payload.get('claims'))
    abstract = _field_text(payload.get('abstract'))
    summary = _field_text(payload.get('summary'))
    parts = tuple(part for part in (claims, abstract, summary) if part)
    return PatentText(
        application_number=_field_text(payload.get('application_number')),
        text=strip_soh_markup(' '.join(parts)),
    )


def json_patent_files(root: Path) -> tuple[Path, ...]:
    """Return regular-file JSON paths under ``root``."""
    return tuple(sorted(path for path in root.rglob('*.json') if path.is_file()))


def sample_json_paths(
    root: Path,
    *,
    limit: int,
    seed: int,
    index_cache: Path | None = None,
) -> tuple[tuple[Path, ...], int]:
    """Return ``limit`` JSON paths sampled without replacement, plus the pool size."""
    pool = (
        tuple(
            Path(line)
            for line in index_cache.read_text(encoding='utf-8').splitlines()
            if line.strip()
        )
        if index_cache is not None and index_cache.is_file()
        else json_patent_files(root)
    )
    if not pool:
        msg = f'No JSON patent files under {root}'
        raise FileNotFoundError(msg)
    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage] - deterministic sampling, not cryptographic
    chosen = tuple(rng.sample(pool, k=min(limit, len(pool))))
    return chosen, len(pool)

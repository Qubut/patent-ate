"""Local Ray session with a short AF_UNIX temp dir."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV'] = '0'

import ray

RAY_TEMP_DIR = Path(os.environ.get('PATENT_ATE_RAY_TEMP', '/tmp/patent-ate-ray'))  # ruff: ignore[hardcoded-temp-file]


def ensure_local_ray() -> bool:
    """Start a local CPU Ray runtime if one is not already up.

    Returns True when this call started Ray and the caller must shut it down.
    Session sockets are AF_UNIX and cannot exceed 107 bytes, so the temp dir
    stays under 107 bytes. The UV runtime-env hook is captured at import; it
    is cleared here before ``ray.init`` so workers are not spawned with
    ``uv run --python``.
    """
    if ray.is_initialized():
        return False
    RAY_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV'] = '0'
    importlib.import_module('ray._private.ray_constants').RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    library_path = os.environ.get('LD_LIBRARY_PATH')
    runtime_env = {'env_vars': {'LD_LIBRARY_PATH': library_path}} if library_path else None
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        num_gpus=0,
        _temp_dir=str(RAY_TEMP_DIR),
        runtime_env=runtime_env,
        _skip_env_hook=True,
    )
    return True

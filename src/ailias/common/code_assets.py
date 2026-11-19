"""Locate source modules for immutable native-worker and provenance snapshots."""

from importlib.util import find_spec
from pathlib import Path


def module_source(module_name: str) -> Path:
    specification = find_spec(module_name)
    if specification is None or specification.origin is None:
        raise ValueError(f"No file-backed module: {module_name}")
    return Path(specification.origin).resolve(strict=True)

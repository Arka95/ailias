"""Walk nested JSON values without coupling callers to report schemas."""

from collections.abc import Iterator
from typing import Any


def iter_nested_strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from iter_nested_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_nested_strings(nested)
    elif isinstance(value, str):
        yield value


def iter_artifact_references(value: Any) -> Iterator[dict]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and "sha256" in value:
            yield value
        for nested in value.values():
            yield from iter_artifact_references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_artifact_references(nested)

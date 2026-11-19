"""Serialize structured values and walk nested JSON without coupling callers to report schemas."""

from collections.abc import Iterator
from dataclasses import fields, is_dataclass
import json
from pathlib import PurePath
from typing import Any
from uuid import UUID


def to_json_value(value: Any) -> Any:
    """Convert structured objects, dataclasses and path/UUID references into strict JSON-compatible values.
    Prefer explicit to_dict() contracts and reject unsupported objects rather than stringifying their contents.
    """
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _json_default(value: Any) -> Any:
    """Adapt one non-native JSON value using its explicit data contract or a standard-library representation.
    Leave unknown runtime objects unsupported so callers must choose how their outputs are persisted.
    """
    serialize = getattr(value, "to_dict", None)
    if callable(serialize):
        return serialize()
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    if isinstance(value, (PurePath, UUID)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}; provide to_dict() or a file reference")


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

"""Generate job IDs and validate path-safe user identifiers."""

from datetime import datetime, timezone
import re
import uuid


def generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]


def validate_identifier(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}", value):
        raise ValueError("Invalid run/group identifier")
    return value

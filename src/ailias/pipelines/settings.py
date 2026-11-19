"""Validate cross-domain photo appearance and native handoff configuration."""

import json
import re

from ailias.settings import PROJECT_ROOT

DEFAULT_APPEARANCE_CONFIG = PROJECT_ROOT / "config" / "appearance.json"
DEFAULT_REVIEW_CONFIG = PROJECT_ROOT / "config" / "character-review.json"


def load_appearance_policy(path=DEFAULT_APPEARANCE_CONFIG):
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not re.fullmatch(r"cuda:[0-9]+", policy.get("device", "")):
        raise ValueError("Photo baking requires an explicitly selected CUDA device")
    for key, low, high in (
        ("texture_size", 256, 4096),
        ("photo_size", 256, 2048),
        ("max_views", 1, 64),
        ("render_samples", 1, 128),
        ("hair_strands", 0, 50000),
        ("hair_segments", 3, 16),
        ("seed", 0, 2**31 - 1),
    ):
        if type(policy.get(key)) is not int or not low <= policy[key] <= high:
            raise ValueError(f"{key} must be an integer in {low}..{high}")
    if policy.get("native_hair_asset") is not None:
        from ailias.metahuman.character_features_worker import native_asset

        native_asset(policy["native_hair_asset"])
    if "assembly" in policy:
        from ailias.metahuman.character_features_worker import (
            validate_assembly_settings,
        )

        validate_assembly_settings(policy["assembly"])
    if "native_appearance" in policy:
        from ailias.metahuman.character_features_worker import (
            validate_native_appearance,
        )

        validate_native_appearance(policy["native_appearance"])
        if not policy.get("native_hair_asset"):
            raise ValueError(
                "Native appearance without photo baking requires native_hair_asset"
            )
    return policy

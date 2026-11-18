"""Validate head fitting, sampled-photo quality and appearance policies."""

import json
import math
import re
from dataclasses import asdict, dataclass

from ailias.settings import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "facefitting.json"
MODEL_SPECS = PROJECT_ROOT / "config" / "facefitting-models.json"
DEFAULT_VIDEO_CONFIG = PROJECT_ROOT / "config" / "video-evidence.json"
DEFAULT_APPEARANCE_CONFIG = PROJECT_ROOT / "config" / "appearance.json"
DEFAULT_REVIEW_CONFIG = PROJECT_ROOT / "config" / "character-review.json"


@dataclass(frozen=True)
class FitPolicy:
    device: str = "cpu"
    max_iterations: int = 800
    identity_candidates: tuple = (48, 96, 170)
    expression_modes: tuple = (12, 12, 24)
    identity_regularization: float = 0.00001
    expression_regularization: float = 0.000005
    max_nfev: int = 55
    identity_coefficient_limit: float = 3.0
    expression_coefficient_limit: float = 3.0
    view_batch_size: int = 16
    focal_factor_min: float = 0.1
    focal_factor_max: float = 5.0

    def __post_init__(self):
        if self.device != "cpu" and (not isinstance(self.device, str)
                                     or not re.fullmatch(r"cuda:[0-9]+", self.device)):
            raise ValueError("device must be explicitly cpu or cuda:N")
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 5000:
            raise ValueError("max_iterations must be an integer in 1..5000")
        if (not self.identity_candidates
                or any(type(v) is not int or not 1 <= v <= 170 for v in self.identity_candidates)):
            raise ValueError("Head identity modes must be integers in 1..170")
        if (len(self.expression_modes) != 3
                or any(type(v) is not int or not 1 <= v <= maximum
                       for v, maximum in zip(self.expression_modes, (100, 100, 150)))):
            raise ValueError("Expression capacities must fit the 100/100/150 regional bases")
        if len(set(self.identity_candidates)) != len(self.identity_candidates):
            raise ValueError("Require three expression regions and distinct identity candidates")
        for value in (self.identity_regularization, self.expression_regularization,
                      self.identity_coefficient_limit, self.expression_coefficient_limit,
                      self.focal_factor_min, self.focal_factor_max):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Fit regularization must be finite and positive")
        if type(self.max_nfev) is not int or not 1 <= self.max_nfev <= 500:
            raise ValueError("max_nfev must be an integer in 1..500")
        if type(self.view_batch_size) is not int or self.view_batch_size < 1:
            raise ValueError("view_batch_size must be a positive integer; it does not limit input views")
        if self.focal_factor_min >= self.focal_factor_max:
            raise ValueError("Focal factor minimum must be below its maximum")

    def record(self):
        return asdict(self)


def load_policy(path=DEFAULT_CONFIG):
    return FitPolicy(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class VideoPolicy:
    frame_mode: str = "sampled"
    sample_fps: float = 1
    max_samples_per_video: int = 5000
    max_duration_seconds: float = 120
    image_long_side: int = 1920
    detection_long_side: int = 1280
    max_identity_frames: int = 24
    minimum_spacing_seconds: float = 1
    validation_guard_seconds: float = 2
    neutral_expression_max: float = .18
    jaw_open_max: float = .1
    blink_max: float = .2
    minimum_eye_span_pixels: float = 100
    minimum_laplacian_variance: float = 20

    def __post_init__(self):
        if not isinstance(self.frame_mode, str) or self.frame_mode not in {"all", "sampled"}:
            raise ValueError("frame_mode must be all or sampled")
        bounds = {
            "sample_fps": (.1, 10), "max_samples_per_video": (1, 20000),
            "max_duration_seconds": (1, 600), "image_long_side": (256, 8192),
            "detection_long_side": (256, 2048),
            "max_identity_frames": (3, 100), "minimum_spacing_seconds": (.1, 10),
            "validation_guard_seconds": (.1, 30), "neutral_expression_max": (0, 1),
            "jaw_open_max": (0, 1), "blink_max": (0, 1),
            "minimum_eye_span_pixels": (10, 2000), "minimum_laplacian_variance": (0, 10000),
        }
        for key, (lower, upper) in bounds.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite numeric")
            if not lower <= value <= upper:
                raise ValueError(f"{key} outside supported range")
        for key in ("max_samples_per_video", "image_long_side", "detection_long_side", "max_identity_frames"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")


def load_appearance_policy(path=DEFAULT_APPEARANCE_CONFIG):
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not re.fullmatch(r"cuda:[0-9]+", policy.get("device", "")):
        raise ValueError("Photo baking requires an explicitly selected CUDA device")
    for key, low, high in (
        ("texture_size", 256, 4096), ("photo_size", 256, 2048), ("max_views", 1, 64),
        ("render_samples", 1, 128), ("hair_strands", 0, 50000),
        ("hair_segments", 3, 16), ("seed", 0, 2**31-1),
    ):
        if type(policy.get(key)) is not int or not low <= policy[key] <= high:
            raise ValueError(f"{key} must be an integer in {low}..{high}")
    if policy.get("native_hair_asset") is not None:
        from ailias.facefitting.unreal.character_features_worker import native_asset

        native_asset(policy["native_hair_asset"])
    if "assembly" in policy:
        from ailias.facefitting.unreal.character_features_worker import validate_assembly_settings

        validate_assembly_settings(policy["assembly"])
    return policy

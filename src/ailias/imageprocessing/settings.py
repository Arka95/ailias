"""Load and validate image-processing thresholds and model/backend configuration."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

from ailias.settings import PROJECT_ROOT

DEFAULT_PROCESSING_CONFIG = PROJECT_ROOT / "config" / "imageprocessing.json"
DEFAULT_FACE_QUALITY_CONFIG = PROJECT_ROOT / "config" / "face_quality.json"
# MediaPipe's model emits 468 face points and 10 additional iris points.
MEDIAPIPE_BASE_FACE_POINT_COUNT = 468
MEDIAPIPE_LANDMARK_COUNT = 478


@dataclass(frozen=True)
class ProcessingPolicy:
    detection_score: float = 0.85
    detection_max_side: int = 1600
    cpu_threads: int = 2
    # Minimum native face-box short side, in pixels; smaller faces are flagged as too small.
    min_face_size: int = 256
    # Minimum grayscale Laplacian variance after downsampling the face to at most 256 pixels per side.
    # Lower variance flags blur or low detail; this is a focus heuristic, not learned confidence.
    min_laplacian_variance: float = 40.0
    # Largest permitted fraction of sampled grayscale face pixels below intensity 16; higher flags underexposure.
    max_dark_fraction: float = 0.5
    # Largest permitted fraction of sampled grayscale face pixels at/above intensity 240; higher flags overexposure.
    max_bright_fraction: float = 0.5
    # Historical manifest fields only: active extraction reads FaceQualityPolicy instead.
    # Keep accepting stored values so legacy CLI/downstream manifests remain loadable.
    min_landmark_alpha: int = 240
    min_supported_landmarks: int = 150
    max_neutral_expression: float = 0.5
    max_pitch_roll_degrees: float = 25.0
    max_yaw_degrees: float = 70.0
    identity_cosine_threshold: float = 0.5
    identity_ambiguity_margin: float = 0.05
    portrait_side_margin: float = 0.6
    portrait_top_margin: float = 0.75
    portrait_bottom_margin: float = 1.0
    curation_accept_cosine: float = 0.5
    curation_reject_cosine: float = 0.6
    curation_reject_positive_ceiling: float = 0.35
    curation_score_margin: float = 0.1

    def __post_init__(self):
        """Validate configurable image-quality and collection thresholds before processing.
        Bound dense-face review settings by their model and pixel-coordinate contracts.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{field.name} must be a non-negative finite number")
        for name in ("detection_max_side", "cpu_threads", "min_face_size",
                     "min_landmark_alpha", "min_supported_landmarks"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.detection_max_side < 320:
            raise ValueError("detection_max_side must be at least 320")
        if self.min_landmark_alpha > 255:
            raise ValueError("min_landmark_alpha must be between 1 and 255")
        if self.min_supported_landmarks > MEDIAPIPE_BASE_FACE_POINT_COUNT:
            raise ValueError("min_supported_landmarks exceeds the number of base facial points")
        if not 0 < self.max_pitch_roll_degrees <= 180:
            raise ValueError("max_pitch_roll_degrees must be between 0 (exclusive) and 180")
        if not 0 < self.max_yaw_degrees <= 90:
            raise ValueError("max_yaw_degrees must be between 0 (exclusive) and 90")
        for name in ("detection_score", "max_dark_fraction", "max_bright_fraction",
                     "max_neutral_expression",
                     "identity_cosine_threshold", "identity_ambiguity_margin",
                     "curation_accept_cosine", "curation_reject_cosine",
                     "curation_reject_positive_ceiling", "curation_score_margin"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.identity_cosine_threshold == 0:
            raise ValueError("identity_cosine_threshold must be positive")
        if not 0 < self.curation_reject_positive_ceiling < self.curation_accept_cosine:
            raise ValueError("Curation reject-positive ceiling must be positive and below accept cosine")
        if self.curation_reject_cosine == 0 or self.curation_score_margin == 0:
            raise ValueError("Curation rejection and margin thresholds must be positive")


def load_policy(path: Path = DEFAULT_PROCESSING_CONFIG) -> ProcessingPolicy:
    """Load image-processing thresholds from the selected JSON configuration file.
    Apply defaults for omitted fields and validate the resulting policy before use.
    """
    return ProcessingPolicy(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class FaceQualityPolicy:
    """Configure quality annotations during feature extraction without identity or angle-rejection rules.
    Thresholds describe possible limitations; they never decide whether grouping keeps an image.
    """

    # Native face-crop short side in pixels; smaller crops request a detail warning.
    min_face_size: int = 256
    # Minimum grayscale Laplacian variance after downsampling to at most 256 pixels per side.
    min_laplacian_variance: float = 40.0
    # Warn above these fractions of grayscale crop pixels below 16 or at/above 240, respectively.
    max_dark_fraction: float = 0.5
    max_bright_fraction: float = 0.5
    # Minimum 1-255 opacity for quality-support counting, independent of raw positive-alpha visibility.
    min_landmark_alpha: int = 240
    # Minimum supported landmarks among the 468 base-face predictions, excluding iris points.
    min_supported_landmarks: int = 150
    # Maximum expression/jaw score before warning; this is not calibrated confidence.
    max_neutral_expression: float = 0.5

    def __post_init__(self):
        """Validate finite warning thresholds and the model's opacity/count domains.
        Reject invalid configuration before inference rather than weakening quality annotations.
        """
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be a nonnegative finite number")
        for name, maximum in (("min_face_size", None), ("min_landmark_alpha", 255),
                              ("min_supported_landmarks", MEDIAPIPE_BASE_FACE_POINT_COUNT)):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1 or (maximum is not None and value > maximum):
                raise ValueError(f"Invalid {name}")
        for name in ("max_dark_fraction", "max_bright_fraction", "max_neutral_expression"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be between 0 and 1")


def load_face_quality_policy(path: str | Path = DEFAULT_FACE_QUALITY_CONFIG) -> FaceQualityPolicy:
    """Load feature-extraction quality settings from the selected JSON configuration file.
    Apply defaults for omitted fields and validate thresholds without identity-related configuration.
    """
    return FaceQualityPolicy(**json.loads(Path(path).read_text(encoding="utf-8")))

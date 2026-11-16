import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

from ailias.settings import PROJECT_ROOT

DEFAULT_PROCESSING_CONFIG = PROJECT_ROOT / "config" / "imageprocessing.json"


@dataclass(frozen=True)
class ProcessingPolicy:
    detection_score: float = 0.85
    detection_max_side: int = 1600
    cpu_threads: int = 2
    min_face_size: int = 256
    min_laplacian_variance: float = 40.0
    max_dark_fraction: float = 0.5
    max_bright_fraction: float = 0.5
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
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{field.name} must be a non-negative finite number")
        for name in ("detection_max_side", "cpu_threads", "min_face_size"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.detection_max_side < 320:
            raise ValueError("detection_max_side must be at least 320")
        for name in ("detection_score", "max_dark_fraction", "max_bright_fraction",
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
    return ProcessingPolicy(**json.loads(path.read_text(encoding="utf-8")))

"""Structured facial evidence and explicit in-memory JSON serialization."""

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class FaceRegionFeatures:
    """Keep colour, coverage and review evidence for one semantic region or combined region.
    Missing colour remains explicit rather than inventing measurements for absent or clipped pixels.
    """

    status: Literal["observed", "review"]
    parser_classes: list[int]
    region_pixels: int
    sample_pixels: int
    bounds_xy: list[list[int]] | None = None
    median_linear_rgb: list[float] | None = None
    reason: str | None = None
    photographed_lighting_retained: bool | None = None
    confidence: str | None = None

    def to_dict(self) -> dict:
        """Return ordinary Python values without absent optional measurements.
        Keep review reasons and parser IDs available even when no colour was measurable.
        """
        return {name: value for name, value in asdict(self).items() if value is not None}

    def __str__(self) -> str:
        """Serialize this region as strict JSON without writing any files.
        Reject nonfinite numeric values rather than emitting nonstandard JSON tokens.
        """
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class FaceFeaturesBiseNet:
    """Retain natural face/head evidence derived from one existing BiSeNet label image.
    Keep individual parts plus the previous combined brow/lip summaries; exclude accessories and clothing.
    """

    skin: FaceRegionFeatures
    left_eyebrow: FaceRegionFeatures
    right_eyebrow: FaceRegionFeatures
    left_eye: FaceRegionFeatures
    right_eye: FaceRegionFeatures
    left_ear: FaceRegionFeatures
    right_ear: FaceRegionFeatures
    nose: FaceRegionFeatures
    mouth_interior: FaceRegionFeatures
    upper_lip: FaceRegionFeatures
    lower_lip: FaceRegionFeatures
    scalp_hair: FaceRegionFeatures
    eyebrows: FaceRegionFeatures
    lips: FaceRegionFeatures

    def to_dict(self) -> dict:
        """Return every named region as a nested dictionary of retained evidence.
        Preserve absent-region review records without exposing dataclass instances.
        """
        return {region.name: getattr(self, region.name).to_dict() for region in fields(self)}

    def __str__(self) -> str:
        """Serialize all retained regions as strict JSON without exporting artifacts.
        Use the same values and structure returned by to_dict().
        """
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class FaceFeatures:
    """Collect per-face predictions, measured photo features and review evidence as attributes.
    Preserve source identity and optional errors without owning images or inference resources.
    """

    status: Literal["ready", "review", "error"] = "review"
    features: dict | None = None
    bisenet: FaceFeaturesBiseNet | None = None
    landmarks: dict | None = None
    blendshapes: dict = field(default_factory=dict)
    facial_transform: list[list[float]] | None = None
    quality: dict = field(default_factory=lambda: {"reasons": []})
    source: Path | str | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    features_colour_source: str = "original_rgb"
    metric_depth_measured: bool = False
    execution: dict = field(default_factory=dict)
    error_type: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        """Return nested Python data with source paths represented as strings.
        Serialize structured BiSeNet regions using their own evidence-preserving contract.
        """
        result = asdict(self)
        if self.source is not None:
            result["source"] = str(self.source)
        if self.bisenet is not None:
            result["bisenet"] = self.bisenet.to_dict()
        return result

    def __str__(self) -> str:
        """Serialize the complete result as strict JSON without creating sidecars.
        Keep single-image, review and batch-error outputs under the same JSON contract.
        """
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

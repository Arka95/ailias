"""Structured facial evidence and explicit in-memory JSON serialization."""

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Literal

from PIL import Image


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
class FaceQuality:
    """Retain image-quality metrics and descriptive warnings without making selection decisions.
    Missing crop metrics remain unknown; support, clipping and expression evidence stay explicit.
    """

    native_short_side: int | None = None
    laplacian_variance: float | None = None
    dark_fraction: float | None = None
    bright_fraction: float | None = None
    supported_landmarks: int | None = None
    clipped_landmarks: int | None = None
    max_expression_score: float | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return quality metrics and warnings as ordinary Python values, including unknown metrics.
        Keep the result independent of source paths, selection policy and image pixels.
        """
        return asdict(self)

    def __str__(self) -> str:
        """Return strict JSON for the same measurements exposed by to_dict().
        Serialize in memory without saving sidecars or accepting nonfinite numeric tokens.
        """
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class FaceFeatures:
    """Keep facial measurements and raw model predictions separate from collection decisions.
    Include extracted orientation/quality; leave source metadata, failures and selection to result records.
    """

    features: dict | None = None
    bisenet: FaceFeaturesBiseNet | None = None
    landmarks: dict | None = None
    blendshapes: dict = field(default_factory=dict)
    facial_transform: list[list[float]] | None = None
    orientation: dict | None = None
    quality: FaceQuality | None = None
    features_colour_source: str = "original_rgb"

    def to_dict(self) -> dict:
        """Return facial measurements as nested ordinary Python data.
        Serialize structured BiSeNet regions using their own evidence-preserving contract.
        """
        result = asdict(self)
        if self.bisenet is not None:
            result["bisenet"] = self.bisenet.to_dict()
        return result

    def __str__(self) -> str:
        """Serialize the complete result as strict JSON without creating sidecars.
        Exclude image pixels, processing metadata and collection decisions from this contract.
        """
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class FaceExtractionResult:
    """Associate facial data with its image, source metadata and any extraction failure.
    Keep pixels available for grouping/review copies without putting them inside FaceFeatures.
    """

    features: FaceFeatures | None = None
    image: Image.Image | None = field(default=None, repr=False)
    head_image: Image.Image | None = field(default=None, repr=False)
    source: Path | str | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    execution: dict = field(default_factory=dict)
    error_type: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        """Serialize provenance, facial data and failures without embedding image pixels.
        Retain source size so stored metadata remains useful without live Pillow objects.
        """
        return {
            "features": self.features.to_dict() if self.features is not None else None,
            "source": str(self.source) if self.source is not None else None,
            "frame_index": self.frame_index, "timestamp_seconds": self.timestamp_seconds,
            "image_size": list(self.image.size) if self.image is not None else None,
            "execution": dict(self.execution), "error_type": self.error_type, "reason": self.reason,
        }

    def __str__(self) -> str:
        """Return metadata and facial data as strict JSON without writing artifacts.
        Omit in-memory image pixels rather than silently stringifying Pillow objects.
        """
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

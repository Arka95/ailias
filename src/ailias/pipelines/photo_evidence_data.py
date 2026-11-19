"""Pixel-free records for freshly prepared photo evidence and explicit artifact exports."""

from dataclasses import dataclass, field, fields
import json

from ailias.common.json_values import to_json_value
from ailias.imageprocessing.face_features import FaceFeatures


class _EvidenceRecord:
    """Serialize workflow evidence without retaining model handles or live image pixels.
    Delegate nested feature serialization to its own typed contract.
    """

    def to_dict(self) -> dict:
        """Return strict JSON values using each nested record's explicit serializer.
        Leave all artifact writes to the workflow that owns them.
        """
        return to_json_value(
            {item.name: getattr(self, item.name) for item in fields(self)}
        )

    def __str__(self) -> str:
        """Return strict JSON in memory without creating a manifest or sidecars.
        Reject unsupported runtime objects instead of silently stringifying them.
        """
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class PhotoEvidenceFrame(_EvidenceRecord):
    """Associate one view's facial evidence with its actual pixel grid and capture provenance.
    Explicit reviewed crops retain their original reference/bounds in capture, never overwrite it.
    """

    view_id: str
    capture: dict
    image: dict
    features: FaceFeatures | None
    regions: dict
    head_image: dict
    selected: bool
    view_group: str
    reason: str | None = None
    error_type: str | None = None
    execution: dict = field(default_factory=dict)
    review_copy: dict | None = None
    review_copy_error: str | None = None


@dataclass
class PhotoEvidenceBatch(_EvidenceRecord):
    """Keep extraction results, model provenance and this job's explicit exported artifacts.
    Source images are reused, but predictions and CNN preparation are fresh.
    """

    schema_version: int = 2
    status: str = "running"
    frames: list[PhotoEvidenceFrame] = field(default_factory=list)
    models: dict[str, dict] = field(default_factory=dict)
    artifacts: dict[str, dict] = field(default_factory=dict)
    sampled_images_reused: bool = True
    previous_predictions_reused: bool = False
    video_decoding_performed: bool = False
    observed_colours_are_intrinsic_albedo: bool = False
    error_type: str | None = None
    error_message: str | None = None
    cleanup_errors: list[str] = field(default_factory=list)

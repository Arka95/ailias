"""Orient, group and select images of one known person; preserve excluded source copies."""

from collections.abc import Iterable
from dataclasses import dataclass, field, fields, replace
import json
import math
from pathlib import Path
import shutil
import uuid

import numpy as np

from ailias.imageprocessing.face_features import FaceExtractionResult
from ailias.imageprocessing.features import face_quality
from ailias.imageprocessing.settings import MEDIAPIPE_BASE_FACE_POINT_COUNT, MEDIAPIPE_LANDMARK_COUNT
from ailias.settings import PROJECT_ROOT


DEFAULT_GROUPING_CONFIG = PROJECT_ROOT / "config" / "face_grouping.json"
# Allow numerical drift after removing uniform scale, not non-rigid shear or reflected coordinates.
ROTATION_VALIDATION_TOLERANCE = 0.02


@dataclass(frozen=True)
class FaceGroupingPolicy:
    """Configure descriptive quality warnings and model-space view bins for one person's images.
    No quality warning or angle threshold discards an otherwise usable extraction.
    """

    # Native face-box short side in pixels; smaller faces get a detail warning, not exclusion.
    min_face_size: int = 256
    # Grayscale focus heuristic on a face crop downsampled to at most 256 pixels per side.
    min_laplacian_variance: float = 40.0
    # Warn above these fractions of face pixels below intensity 16 or at/above 240, respectively.
    max_dark_fraction: float = 0.5
    max_bright_fraction: float = 0.5
    # Opacity from 1 to 255 needed to count a landmark as well supported by the prepared head.
    min_landmark_alpha: int = 240
    # Fewer supported base-face points warn; iris points are excluded from this count.
    min_supported_landmarks: int = 150
    # Expression/jaw score above this value warns about expression, not identity or image validity.
    max_neutral_expression: float = 0.5
    # Absolute yaw up to this angle belongs to the frontal view bin.
    frontal_yaw_degrees: float = 20.0
    # Absolute yaw at/above this angle belongs to a profile rather than an oblique bin.
    profile_yaw_degrees: float = 55.0
    # Absolute pitch above this angle adds an up/down bin, including combined side-and-up/down views.
    vertical_pitch_degrees: float = 20.0

    def __post_init__(self):
        """Validate view boundaries, opacity/count domains and finite quality-warning thresholds.
        Reject invalid configuration before grouping or writing any review copies.
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
        if not 0 < self.frontal_yaw_degrees < self.profile_yaw_degrees <= 90:
            raise ValueError("Yaw bins must satisfy 0 < frontal < profile <= 90 degrees")
        if not 0 < self.vertical_pitch_degrees <= 90:
            raise ValueError("Vertical pitch boundary must be between 0 and 90 degrees")


@dataclass(frozen=True)
class GroupedFaceImage:
    """Associate one extraction with orientation, quality annotations and its selection decision.
    Keep review-copy failures explicit without modifying or replacing original files.
    """

    extraction: FaceExtractionResult
    orientation: dict
    selected: bool = False
    quality: dict = field(default_factory=lambda: {"reasons": []})
    exclusion_reason: str | None = None
    review_copy: Path | None = None
    review_copy_error: str | None = None

    def to_dict(self) -> dict:
        """Return source/feature metadata and grouping decisions without image pixel objects.
        Convert an optional review-copy path into a JSON-compatible string.
        """
        return {
            "extraction": self.extraction.to_dict(), "orientation": dict(self.orientation),
            "selected": self.selected, "quality": self.quality, "exclusion_reason": self.exclusion_reason,
            "review_copy": str(self.review_copy) if self.review_copy is not None else None,
            "review_copy_error": self.review_copy_error,
        }


@dataclass(frozen=True)
class FaceGroupingResult:
    """Keep all view groups, including excluded candidates and their explicit decisions.
    Expose selected/excluded lists without creating a second independent set of decisions.
    """

    groups: dict[str, list[GroupedFaceImage]]

    @property
    def selected(self) -> list[GroupedFaceImage]:
        """Return every usable image across all view groups without quotas or ranking cutoffs.
        Preserve side, upper, lower and unknown-orientation views when extraction is usable.
        """
        return [item for group in self.groups.values() for item in group if item.selected]

    @property
    def excluded(self) -> list[GroupedFaceImage]:
        """Return failed or invalid extractions with their reasons and optional review-copy references.
        Never interpret descriptive quality warnings as exclusions.
        """
        return [item for group in self.groups.values() for item in group if not item.selected]

    def to_dict(self) -> dict:
        """Serialize every group and its decisions as nested ordinary Python values.
        Keep source references and retained features without embedding image bytes.
        """
        return {"groups": {name: [item.to_dict() for item in group] for name, group in self.groups.items()}}

    def __str__(self) -> str:
        """Return the grouping report as strict JSON without saving a sidecar automatically.
        Explicit review-image copying is the only filesystem output of grouping.
        """
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


def load_grouping_policy(path: str | Path = DEFAULT_GROUPING_CONFIG) -> FaceGroupingPolicy:
    """Load the single-person grouping configuration and validate its named thresholds.
    Explicit caller policies bypass this file without introducing environment-variable overrides.
    """
    return FaceGroupingPolicy(**json.loads(Path(path).read_text(encoding="utf-8")))


def determine_face_orientation(transform, policy: FaceGroupingPolicy) -> dict:
    """Derive approximate model-space Euler angles from MediaPipe's raw face transform.
    Remove uniform scale before validating rotation; unavailable orientation remains an unknown view.
    """
    unknown = {"category": "unknown", "calibrated": False, "reason": "missing_or_invalid_pose"}
    try:
        matrix = np.asarray(transform, dtype=float)
    except (TypeError, ValueError):
        return unknown
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return unknown
    linear = matrix[:3, :3]
    scale = float(np.linalg.norm(linear[:, 0]))
    if scale <= np.finfo(float).eps:
        return unknown
    # MediaPipe's canonical-to-runtime transform can contain uniform scale as well as rotation.
    rotation = linear / scale
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=ROTATION_VALIDATION_TOLERANCE)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=ROTATION_VALIDATION_TOLERANCE)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=ROTATION_VALIDATION_TOLERANCE)):
        return unknown
    yaw = math.degrees(math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0])))
    pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return {"category": _view_category(yaw, pitch, policy), "pitch_deg": pitch,
            "yaw_deg": yaw, "roll_deg": roll, "calibrated": False}


def _view_category(yaw: float, pitch: float, policy: FaceGroupingPolicy) -> str:
    """Label approximate yaw and pitch bins without declaring any view invalid.
    Preserve combined side-and-vertical views rather than collapsing their upper/lower information.
    """
    if abs(yaw) <= policy.frontal_yaw_degrees:
        category = "front"
    else:
        side = "left" if yaw < 0 else "right"
        category = side + ("_profile" if abs(yaw) >= policy.profile_yaw_degrees else "_oblique")
    if abs(pitch) > policy.vertical_pitch_degrees:
        vertical = "up" if pitch > 0 else "down"
        category = vertical if category == "front" else category + "_" + vertical
    return category


def _face_bounds(extraction: FaceExtractionResult) -> tuple[list | None, np.ndarray | None, str | None]:
    """Check that extracted base-face coordinates overlap their source image with nonzero extent.
    Treat missing, nonfinite or entirely outside landmarks as invalid, not as another identity.
    """
    if extraction.error_type is not None or extraction.reason is not None:
        return None, None, extraction.reason or extraction.error_type
    if extraction.image is None or extraction.features is None or extraction.features.landmarks is None:
        return None, None, "missing_image_or_face_features"
    try:
        points = np.asarray(extraction.features.landmarks["source_xy"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None, None, "invalid_dense_landmarks"
    if points.shape != (MEDIAPIPE_LANDMARK_COUNT, 2) or not np.isfinite(points).all():
        return None, None, "invalid_dense_landmarks"
    base = points[:MEDIAPIPE_BASE_FACE_POINT_COUNT]
    low, high = np.maximum(0, base.min(0)), np.minimum(extraction.image.size, base.max(0))
    if np.any(high <= low):
        return None, None, "face_outside_image_or_degenerate_landmarks"
    return [*low.tolist(), *(high - low).tolist()], points, None


def _assess_image_quality(extraction: FaceExtractionResult, box: list, points, policy: FaceGroupingPolicy) -> dict:
    """Annotate detail, exposure, clipping, alpha support and expression without rejecting usable views.
    Orientation and facial angles never add an extreme-pose penalty to these warnings.
    """
    quality = face_quality(extraction.image, box, policy)
    base = points[:MEDIAPIPE_BASE_FACE_POINT_COUNT]
    inside = ((base >= 0) & (base < extraction.image.size)).all(1)
    if not inside.all():
        quality["reasons"].append("face_clipped")
    if extraction.head_image is not None and extraction.head_image.size == extraction.image.size:
        if "A" in extraction.head_image.getbands():
            samples = np.clip(np.floor(base).astype(int), 0, np.asarray(extraction.image.size) - 1)
            alpha = np.asarray(extraction.head_image.getchannel("A"))
            supported = inside & (alpha[samples[:, 1], samples[:, 0]] >= policy.min_landmark_alpha)
            if int(supported.sum()) < policy.min_supported_landmarks:
                quality["reasons"].append("insufficient_head_support")
    expression = max((value for name, value in extraction.features.blendshapes.items()
                      if name.startswith(("mouth", "brow", "cheek", "noseSneer", "eyeWide", "eyeSquint"))
                      or name == "jawOpen"), default=0.0)
    if expression > policy.max_neutral_expression:
        quality["reasons"].append("non_neutral_expression")
    return quality


def _copy_excluded_image(extraction: FaceExtractionResult, directory: Path) -> Path:
    """Copy an excluded source file or export an excluded in-memory/video frame into the review directory.
    Allocate an exclusive destination, preserve original bytes for source photos, and clean up partial copies.
    """
    source = Path(extraction.source) if extraction.source is not None else None
    copy_source = source is not None and extraction.frame_index is None and source.is_file()
    if not copy_source and extraction.image is None:
        raise FileNotFoundError("Excluded image has no available source file or frame pixels")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (uuid.uuid4().hex + ((source.suffix or ".bin") if copy_source else ".png"))
    created = False
    try:
        with target.open("xb") as output:
            created = True
            if copy_source:
                with source.open("rb") as original:
                    shutil.copyfileobj(original, output)
            else:
                extraction.image.save(output, format="PNG")
    except (OSError, ValueError):
        if created:
            target.unlink(missing_ok=True)
        raise
    return target


def _select_group_member(item: GroupedFaceImage, directory: Path, policy: FaceGroupingPolicy) -> GroupedFaceImage:
    """Keep every usable extraction and copy only failed/invalid inputs for review.
    Report unavailable copies explicitly rather than deleting originals or silently dropping candidates.
    """
    box, points, reason = _face_bounds(item.extraction)
    if reason is None:
        quality = _assess_image_quality(item.extraction, box, points, policy)
        return replace(item, selected=True, quality=quality)
    excluded = replace(item, exclusion_reason=reason)
    try:
        return replace(excluded, review_copy=_copy_excluded_image(item.extraction, directory))
    except (OSError, ValueError) as error:
        return replace(excluded, review_copy_error=f"{type(error).__name__}: {error}")


def group_face_images(
    extractions: Iterable[FaceExtractionResult], *, review_directory: str | Path,
    policy: FaceGroupingPolicy | None = None,
) -> FaceGroupingResult:
    """Determine orientation, form one person's view groups, then select every usable member.
    Copy excluded candidates into the explicit review directory without modifying or deleting originals.
    """
    policy = policy if policy is not None else load_grouping_policy()
    groups = {}
    for extraction in extractions:
        transform = extraction.features.facial_transform if extraction.features is not None else None
        orientation = determine_face_orientation(transform, policy)
        groups.setdefault(orientation["category"], []).append(GroupedFaceImage(extraction, orientation))
    directory = Path(review_directory)
    return FaceGroupingResult({
        name: [_select_group_member(item, directory, policy) for item in members]
        for name, members in groups.items()
    })

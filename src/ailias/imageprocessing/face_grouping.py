"""Group extracted orientations and select one person's images; preserve excluded source copies."""

from collections.abc import Iterable
from dataclasses import dataclass, fields, replace
import json
import math
from pathlib import Path
import shutil
import uuid

import numpy as np

from ailias.imageprocessing.face_features import FaceExtractionResult, FaceQuality
from ailias.imageprocessing.settings import MEDIAPIPE_BASE_FACE_POINT_COUNT, MEDIAPIPE_LANDMARK_COUNT
from ailias.settings import PROJECT_ROOT


DEFAULT_GROUPING_CONFIG = PROJECT_ROOT / "config" / "face_grouping.json"


@dataclass(frozen=True)
class FaceGroupingPolicy:
    """Configure view bins using orientation already calculated by feature extraction.
    View boundaries never discard an otherwise usable extraction.
    """

    # Absolute yaw up to this angle belongs to the frontal view bin.
    frontal_yaw_degrees: float = 20.0
    # Absolute yaw at/above this angle belongs to a profile rather than an oblique bin.
    profile_yaw_degrees: float = 55.0
    # Absolute pitch above this angle adds an up/down bin, including combined side-and-up/down views.
    vertical_pitch_degrees: float = 20.0

    def __post_init__(self):
        """Validate finite, ordered yaw and pitch boundaries for view classification.
        Reject invalid configuration before grouping or writing any review copies.
        """
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be a nonnegative finite number")
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
    selected: bool = False
    exclusion_reason: str | None = None
    review_copy: Path | None = None
    review_copy_error: str | None = None

    @property
    def orientation(self) -> dict:
        """Expose the extracted orientation without computing another pose or storing a duplicate.
        Report unavailable orientation for failed or partially extracted facial evidence.
        """
        features = self.extraction.features
        if features is not None and features.orientation is not None:
            return features.orientation
        return {"calibrated": False, "reason": "missing_or_invalid_pose"}

    @property
    def quality(self) -> FaceQuality | None:
        """Expose the same structured quality record produced by feature extraction.
        Leave unavailable assessment as None instead of remeasuring pixels during grouping.
        """
        return self.extraction.features.quality if self.extraction.features is not None else None

    def to_dict(self) -> dict:
        """Return source/feature metadata and grouping decisions without image pixel objects.
        Convert an optional review-copy path into a JSON-compatible string.
        """
        return {
            "extraction": self.extraction.to_dict(), "orientation": dict(self.orientation),
            "selected": self.selected, "quality": self.quality.to_dict() if self.quality is not None else None,
            "exclusion_reason": self.exclusion_reason,
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


def _view_category(orientation: dict, policy: FaceGroupingPolicy) -> str:
    """Label approximate yaw and pitch bins without declaring any view invalid.
    Preserve combined side-and-vertical views rather than collapsing their upper/lower information.
    """
    if "reason" in orientation:
        return "unknown"
    yaw, pitch = orientation["yaw_deg"], orientation["pitch_deg"]
    if abs(yaw) <= policy.frontal_yaw_degrees:
        category = "front"
    else:
        side = "left" if yaw < 0 else "right"
        category = side + ("_profile" if abs(yaw) >= policy.profile_yaw_degrees else "_oblique")
    if abs(pitch) > policy.vertical_pitch_degrees:
        vertical = "up" if pitch > 0 else "down"
        category = vertical if category == "front" else category + "_" + vertical
    return category


def _extraction_invalid_reason(extraction: FaceExtractionResult) -> str | None:
    """Check that extracted base-face coordinates overlap their source image with nonzero extent.
    Treat missing, nonfinite or entirely outside landmarks as invalid, not as another identity.
    """
    if extraction.error_type is not None or extraction.reason is not None:
        return extraction.reason or extraction.error_type
    if extraction.image is None or extraction.features is None or extraction.features.landmarks is None:
        return "missing_image_or_face_features"
    try:
        points = np.asarray(extraction.features.landmarks["source_xy"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return "invalid_dense_landmarks"
    if points.shape != (MEDIAPIPE_LANDMARK_COUNT, 2) or not np.isfinite(points).all():
        return "invalid_dense_landmarks"
    base = points[:MEDIAPIPE_BASE_FACE_POINT_COUNT]
    low, high = np.maximum(0, base.min(0)), np.minimum(extraction.image.size, base.max(0))
    if np.any(high <= low):
        return "face_outside_image_or_degenerate_landmarks"
    return None


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


def _select_group_member(item: GroupedFaceImage, directory: Path) -> GroupedFaceImage:
    """Keep every usable extraction and copy only failed/invalid inputs for review.
    Report unavailable copies explicitly rather than deleting originals or silently dropping candidates.
    """
    reason = _extraction_invalid_reason(item.extraction)
    if reason is None:
        return replace(item, selected=True)
    excluded = replace(item, exclusion_reason=reason)
    try:
        return replace(excluded, review_copy=_copy_excluded_image(item.extraction, directory))
    except (OSError, ValueError) as error:
        return replace(excluded, review_copy_error=f"{type(error).__name__}: {error}")


def group_face_images(
    extractions: Iterable[FaceExtractionResult], *, review_directory: str | Path,
    policy: FaceGroupingPolicy | None = None,
) -> FaceGroupingResult:
    """Consume extracted orientation/quality, form view groups, then select every usable member.
    Copy excluded candidates into the explicit review directory without modifying or deleting originals.
    """
    policy = policy if policy is not None else load_grouping_policy()
    groups = {}
    for extraction in extractions:
        item = GroupedFaceImage(extraction)
        groups.setdefault(_view_category(item.orientation, policy), []).append(item)
    directory = Path(review_directory)
    return FaceGroupingResult({
        name: [_select_group_member(item, directory) for item in members]
        for name, members in groups.items()
    })

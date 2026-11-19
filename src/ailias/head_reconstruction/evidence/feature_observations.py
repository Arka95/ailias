"""Adapt retained facial predictions into numerical fitting observations without inference."""

from collections.abc import Mapping

import numpy as np

from ailias.head_reconstruction.reconstruction_data import HeadViewObservation
from ailias.imageprocessing.face_features import FaceFeatures
from ailias.imageprocessing.settings import (
    MEDIAPIPE_BASE_FACE_POINT_COUNT,
    MEDIAPIPE_LANDMARK_COUNT,
)


def observations_from_face_features(
    features_by_view: Mapping[str, FaceFeatures],
) -> list[HeadViewObservation]:
    """Validate one known person's retained features and preserve caller-supplied view identifiers.
    Convert pixel coordinates/support only; never load images, run models or interpret predicted z as depth.
    """
    observations = []
    for view_id, features in features_by_view.items():
        if not isinstance(view_id, str) or not view_id:
            raise ValueError("View identifiers must be nonempty strings")
        try:
            observation = _observation_from_features(view_id, features)
        except ValueError as error:
            observations.append(
                HeadViewObservation(view_id, "review", reason=str(error))
            )
        else:
            observations.append(observation)
    return observations


def _observation_from_features(
    view_id: str, features: FaceFeatures
) -> HeadViewObservation:
    """Require finite native-grid points, dimensions and boolean foreground support for fitting.
    Derive eye span and suppress out-of-image support without applying image-quality or orientation gates.
    """
    if not isinstance(features, FaceFeatures):
        raise ValueError("missing_face_features")
    try:
        size = np.asarray(features.source_size)
        valid_size = (
            size.shape == (2,)
            and size.dtype.kind in "iu"
            and np.all(size > 0)
            and not any(
                isinstance(value, (bool, np.bool_)) for value in features.source_size
            )
        )
    except (TypeError, ValueError, OverflowError):
        valid_size = False
    if not valid_size:
        raise ValueError("missing_or_invalid_source_size")
    try:
        points = np.asarray(features.landmarks["source_xy"], dtype=float)
        supported = np.asarray(features.landmarks["on_head"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("missing_or_invalid_landmark_geometry") from error
    if points.shape != (MEDIAPIPE_LANDMARK_COUNT, 2) or not np.isfinite(points).all():
        raise ValueError("invalid_dense_landmarks")
    if supported.shape != (MEDIAPIPE_LANDMARK_COUNT,) or supported.dtype.kind != "b":
        raise ValueError("invalid_landmark_support")
    base = points[:MEDIAPIPE_BASE_FACE_POINT_COUNT]
    if np.any(np.minimum(size, base.max(0)) <= np.maximum(0, base.min(0))):
        raise ValueError("face_outside_image_or_degenerate_landmarks")
    eye_span = float(np.linalg.norm(points[33] - points[263]))
    if not np.isfinite(eye_span) or eye_span <= 0:
        raise ValueError("degenerate_outer_eye_span")
    inside = ((points >= 0) & (points < size)).all(1)
    return HeadViewObservation(
        view_id,
        "ready",
        source_size=tuple(size.tolist()),
        source_xy=points.copy(),
        on_head=supported & inside,
        eye_span_source=eye_span,
    )

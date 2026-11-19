"""Synthetic, pixel-grid-consistent facial evidence for workflow boundary tests."""

import numpy as np
from PIL import Image

from ailias.imageprocessing.appearance_features import extract_bisenet_features
from ailias.imageprocessing.face_features import FaceFeatures, FaceQuality


def make_photo_features(size, *, appearance=None, warnings=(), yaw=0, pitch=0):
    """Provide retained source-grid predictions and measured synthetic-region colour without inference.
    Keep fixtures typed so orchestration tests cannot accidentally preserve retired evidence dictionaries.
    """
    points = np.column_stack(
        (
            np.linspace(0.2, 0.8, 478) * size[0],
            (0.4 + 0.1 * np.sin(np.linspace(0, 4, 478))) * size[1],
        )
    )
    image = Image.new("RGB", size, (100, 80, 60))
    labels = Image.new("L", size, 1)
    try:
        regions = extract_bisenet_features(image, labels)
    finally:
        image.close()
        labels.close()
    return FaceFeatures(
        source_size=size,
        landmarks={"source_xy": points.tolist(), "on_head": [True] * 478},
        bisenet=regions,
        features={} if appearance is None else appearance,
        orientation={
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": 0,
            "calibrated": False,
        },
        quality=FaceQuality(supported_landmarks=468, reasons=list(warnings)),
    )

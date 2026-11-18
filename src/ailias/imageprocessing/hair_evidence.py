"""Landmark-region darkness evidence, not a beard segmentation model or measured reflectance."""

import numpy as np
from PIL import Image, ImageDraw


REGIONS = {
    "moustache": (98, 2, 327, 291, 0, 61),
    "beard": (234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
              377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
              291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61),
}


def facial_hair_evidence(rgb, points, labels, skin_linear_rgb):
    """Keep ambiguous shadows explicit; do not relabel the parser's skin class as ground truth."""
    rgb, points, labels = np.asarray(rgb), np.asarray(points, dtype=float), np.asarray(labels)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
            or points.shape != (478, 2) or not np.isfinite(points).all()
            or labels.shape != rgb.shape[:2]):
        raise ValueError("Facial hair evidence requires RGB, same-grid labels and finite 478-point landmarks")
    reference = np.asarray(skin_linear_rgb, dtype=float)
    if (reference.shape != (3,) or not np.isfinite(reference).all()
            or np.any(reference < 0) or np.any(reference > 1)):
        raise ValueError("Facial hair evidence requires a finite skin-colour reference")
    result = {
        "method": "landmark_roi_darkness_relative_to_observed_skin",
        "confidence": "heuristic; shadows and pigmentation can resemble facial hair",
        "distinct_parser_class": False, "intrinsic_reflectance_measured": False, "regions": {},
    }
    for name, indices in REGIONS.items():
        mask = Image.new("L", (rgb.shape[1], rgb.shape[0]))
        ImageDraw.Draw(mask).polygon([tuple(point) for point in points[list(indices)]], fill=255)
        pixels = rgb[(np.asarray(mask) != 0) & (labels == 1)]
        pixels = pixels[pixels.max(1) < 250]
        count = len(pixels)
        if count < 32:
            result["regions"][name] = {"status": "unresolved", "reason": "insufficient_visible_skin_roi"}
            continue
        values = pixels.astype(float)/255
        linear = np.where(values <= .04045, values/12.92, ((values+.055)/1.055)**2.4)
        luminance = linear @ [.2126, .7152, .0722]
        dark = luminance < float(reference @ [.2126, .7152, .0722])*.45
        dark_count = int(dark.sum())
        fraction = dark_count/count
        result["regions"][name] = {
            "status": "estimated", "roi_pixels": count, "dark_pixels": dark_count,
            "dark_fraction": fraction, "presence_estimate": dark_count >= 32 and fraction >= .12,
            "median_dark_linear_rgb": np.median(linear[dark], axis=0).tolist() if dark_count else None,
        }
    return result

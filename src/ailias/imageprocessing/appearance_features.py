"""Measure photo-space shape and visible colours without inventing hidden anatomy."""

from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from ailias.imageprocessing.face_features import FaceFeaturesBiseNet, FaceRegionFeatures


IRIS_EYES = (
    (468, (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)),
    (473, (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)),
)


def iris_colours(rgb, landmarks_xy):
    """Sample a pupil-excluding iris annulus clipped to each observed eyelid."""
    rgb, points = np.asarray(rgb), np.asarray(landmarks_xy, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Iris sampling requires an RGB uint8 image")
    if points.shape != (478, 2) or not np.isfinite(points).all():
        raise ValueError("Iris sampling requires 478 finite source-pixel landmarks")
    height, width = rgb.shape[:2]
    samples = []
    for iris_index, eyelid_indices in IRIS_EYES:
        iris = points[iris_index:iris_index+5]
        center = iris[0]
        axes = np.stack(((iris[1]-iris[3])/2, (iris[2]-iris[4])/2), axis=1)
        if np.linalg.svd(axes, compute_uv=False).min() < 2:
            raise ValueError("Visible iris is too small or degenerate for colour evidence")
        low = np.maximum(0, np.floor(iris.min(0)-1).astype(int))
        high = np.minimum([width, height], np.ceil(iris.max(0)+2).astype(int))
        if np.any(high <= low):
            raise ValueError("Iris lies outside the image")
        columns, rows = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]))
        relative = np.stack((columns+.5-center[0], rows+.5-center[1]), axis=-1)
        radius = np.linalg.norm(relative @ np.linalg.inv(axes).T, axis=-1)
        mask = Image.new("L", tuple((high-low).tolist()))
        ImageDraw.Draw(mask).polygon(
            [tuple(point-low) for point in points[list(eyelid_indices)]], fill=255)
        crop = rgb[low[1]:high[1], low[0]:high[0]]
        valid = ((radius >= .45) & (radius <= .85) & (np.asarray(mask) != 0)
                 & (crop.max(axis=-1) < 250) & (crop.min(axis=-1) > 2))
        pixels = crop[valid].astype(float)/255
        if len(pixels) < 12:
            raise ValueError("Insufficient unobstructed iris pixels")
        linear = np.where(pixels <= .04045, pixels/12.92, ((pixels+.055)/1.055)**2.4)
        luminance = linear @ [.2126, .7152, .0722]
        lower, upper = np.quantile(luminance, [.1, .9])
        retained = linear[(luminance >= lower) & (luminance <= upper)]
        if len(retained) < 8:
            raise ValueError("Insufficient iris pixels after highlight/shadow trimming")
        samples.append({
            "iris_center_landmark": iris_index, "sample_pixels": len(retained),
            "bounds_xy": [low.tolist(), high.tolist()],
            "median_linear_rgb": np.median(retained, axis=0).tolist(),
            "photographed_lighting_retained": True,
        })
    return {"eyes": samples, "median_linear_rgb": np.median(
        [sample["median_linear_rgb"] for sample in samples], axis=0).tolist()}


REGION_LABELS = {
    # Trained BiSeNet class IDs; accessories, background, neck and clothing are not natural face regions.
    "skin": (1,), "left_eyebrow": (2,), "right_eyebrow": (3,),
    "left_eye": (4,), "right_eye": (5,), "left_ear": (7,), "right_ear": (8,),
    "nose": (10,), "mouth_interior": (11,), "upper_lip": (12,), "lower_lip": (13,),
    "scalp_hair": (17,), "eyebrows": (2, 3), "lips": (12, 13),
}
# Existing 8-bit sampling rules: exclude a pixel if any channel is <= 2 or >= 250.
REGION_CHANNEL_FLOOR = 2
REGION_CHANNEL_CEILING = 250
# Fewer usable pixels request review instead of reporting an unstable regional median.
MIN_REGION_COLOUR_SAMPLES = 32
BISENET_CLASS_COUNT = 19
MEASUREMENTS = {
    "outer_eye_span": (33, 263), "inner_eye_span": (133, 362),
    "face_width": (234, 454), "forehead_to_chin": (10, 152),
    "nose_width": (98, 327), "mouth_width": (61, 291),
    "eye_opening_33": (159, 145), "eye_opening_263": (386, 374),
}


def extract_bisenet_features(rgb, parts) -> FaceFeaturesBiseNet:
    """Measure every natural face region using existing BiSeNet labels and source colour pixels.
    This performs pixel statistics only, not another BiSeNet CNN pass or landmark inference.
    """
    rgb, labels = np.asarray(rgb), np.asarray(parts)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
        raise ValueError("BiSeNet features require a nonempty RGB uint8 image")
    if (labels.shape != rgb.shape[:2] or labels.dtype != np.uint8
            or np.any(labels >= BISENET_CLASS_COUNT)):
        raise ValueError("BiSeNet features require native-grid 19-class parser labels")
    return FaceFeaturesBiseNet(**{
        name: _measure_region_features(rgb, labels, classes) for name, classes in REGION_LABELS.items()
    })


def _measure_region_features(rgb, labels, classes: tuple[int, ...]) -> FaceRegionFeatures:
    """Retain coverage, bounds and unclipped linear-RGB colour for one semantic region.
    Keep missing or insufficient colour evidence as an explicit review record with its available bounds.
    """
    mask = np.isin(labels, classes)
    pixels = rgb[mask]
    usable = pixels[(pixels.max(1) < REGION_CHANNEL_CEILING) & (pixels.min(1) > REGION_CHANNEL_FLOOR)]
    locations = np.argwhere(mask)
    result = FaceRegionFeatures(
        status="review", parser_classes=list(classes), region_pixels=len(pixels), sample_pixels=len(usable),
        bounds_xy=([locations.min(0)[::-1].tolist(), (locations.max(0)[::-1] + 1).tolist()]
                   if len(locations) else None),
    )
    if len(usable) < MIN_REGION_COLOUR_SAMPLES:
        return replace(result, reason="insufficient_unclipped_region_pixels")
    values = usable.astype(float) / 255
    linear = np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)
    return replace(
        result, status="observed", median_linear_rgb=np.median(linear, axis=0).tolist(),
        photographed_lighting_retained=True, confidence="uncalibrated_semantic_prediction",
    )


def photo_features(rgb, landmarks_xy, parts, on_head, *, region_features: FaceFeaturesBiseNet | None = None):
    """Keep per-photo regional evidence; colour and dimensions are not intrinsic or calibrated 3D.
    Reuse supplied structured region measurements or derive them from the existing label image.
    """
    rgb, points, labels = np.asarray(rgb), np.asarray(landmarks_xy, dtype=float), np.asarray(parts)
    supported = np.asarray(on_head)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
        raise ValueError("Photo features require a nonempty RGB uint8 image")
    if points.shape != (478, 2) or not np.isfinite(points).all():
        raise ValueError("Photo features require 478 finite pixel landmarks")
    if labels.shape != rgb.shape[:2] or labels.dtype != np.uint8 or np.any(labels >= BISENET_CLASS_COUNT):
        raise ValueError("Photo features require native-grid 19-class parser labels")
    if supported.shape != (478,) or supported.dtype != np.bool_:
        raise ValueError("Photo features require 478 boolean head-support flags")
    height, width = rgb.shape[:2]
    supported = supported & ((points >= 0) & (points < [width, height])).all(1)
    span = float(np.linalg.norm(points[33]-points[263]))
    geometry = {}
    for name, (first, second) in MEASUREMENTS.items():
        if span < 12 or not supported[[33, 263, first, second]].all():
            geometry[name] = {"status": "review", "reason": "insufficient_visible_landmark_support"}
        else:
            length = float(np.linalg.norm(points[first]-points[second]))
            geometry[name] = {
                "status": "observed", "landmark_indices": [first, second],
                "pixels": length, "outer_eye_span_ratio": length/span,
            }
    regions = region_features if region_features is not None else extract_bisenet_features(rgb, labels)
    try:
        indices = [index for center, lid in IRIS_EYES
                   for index in (*range(center, center+5), *lid)]
        if not supported[indices].all():
            raise ValueError("Iris/eyelid landmarks lack visible head support")
        iris = {"status": "observed", **iris_colours(rgb, points)}
    except ValueError as error:
        iris = {"status": "review", "reason": str(error)}
    return {
        "geometry_2d": geometry, "regions": regions.to_dict(), "iris": iris,
        "unresolved": {name: "no_distinct_class_in_current_parser"
                       for name in ("beard", "moustache", "eyelashes")},
        "metric_depth_measured": False, "intrinsic_colour_calibrated": False,
        "visibility": "head-mask support is not independent per-feature occlusion detection",
    }

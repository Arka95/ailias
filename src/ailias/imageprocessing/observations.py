"""Reversible analysis alignment and uncalibrated, subject-relative lighting evidence."""

import math

from PIL import Image


def align_head(cutout, landmarks, side=512):
    import cv2
    import numpy as np

    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return None, {"status": "unavailable", "reason": "missing_or_invalid_five_landmarks"}
    vector = points[1] - points[0]
    distance = float(np.linalg.norm(vector))
    bounds = cutout.getchannel("A").getbbox()
    if distance < 4 or bounds is None:
        return None, {"status": "unavailable", "reason": "insufficient_eye_baseline_or_foreground"}
    midpoint = points[:2].mean(0)
    cosine, sine = vector/distance
    rotation = np.array([[cosine, sine], [-sine, cosine]])
    target = np.array([side*.5, side*.4])
    corners = np.array([[bounds[0], bounds[1]], [bounds[2], bounds[1]],
                        [bounds[0], bounds[3]], [bounds[2], bounds[3]]])
    extents = (corners - midpoint) @ rotation.T
    scale = side*.26/distance
    for axis in (0, 1):
        for value in extents[:, axis]:
            if abs(value) > 1e-8:
                available = side*.95 - target[axis] if value > 0 else target[axis] - side*.05
                scale = min(scale, available/abs(value))
    transform = np.eye(3)
    transform[:2, :2] = scale*rotation
    transform[:2, 2] = target - transform[:2, :2] @ midpoint
    values = np.asarray(cutout, dtype=np.float32)/255
    values[:, :, :3] *= values[:, :, 3:]
    aligned = cv2.warpAffine(values, transform[:2], (side, side), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned[:, :, :3] /= np.maximum(aligned[:, :, 3:], 1e-8)
    aligned[aligned[:, :, 3] < .5/255, :3] = 0
    mapped = points @ transform[:2, :2].T + transform[:2, 2]
    return Image.fromarray(np.rint(np.clip(aligned, 0, 1)*255).astype(np.uint8)), {
        "status": "ready", "method": "eye-axis-similarity-alpha-fit-v1",
        "source_size": list(cutout.size), "size": [side, side],
        "portrait_to_aligned": transform.tolist(),
        "aligned_to_portrait": np.linalg.inv(transform).tolist(),
        "landmarks_5_aligned": mapped.tolist(), "eye_roll_deg": math.degrees(math.atan2(sine, cosine)),
        "scale": scale, "analysis_only": True, "yaw_pitch_frontalized": False,
        "interpolation": "bilinear premultiplied-alpha; transparent padding",
        "scale_constraint": "target eye spacing capped to preserve all observed head foreground",
    }


def face_lighting(image, parts, alpha, bounds):
    import numpy as np

    x, y, width, height = bounds
    box = (max(0, int(x)), max(0, int(y)),
           min(image.width, math.ceil(x + width)), min(image.height, math.ceil(y + height)))
    record = {"method": "visible-skin-srgb-relative-chromaticity-v1",
              "calibrated": False, "intrinsic_albedo": False}
    if box[2] <= box[0] or box[3] <= box[1]:
        return {**record, "skin_pixels": 0, "status": "insufficient_visible_skin",
                "warnings": [], "baseline_penalty": 0.}
    size = (128, 128)
    values = np.asarray(image.crop(box).resize(size, Image.Resampling.BILINEAR), dtype=np.float32)/255
    skin = np.asarray(parts.crop(box).resize(size, Image.Resampling.NEAREST)) == 1
    skin &= np.asarray(alpha.crop(box).resize(size, Image.Resampling.NEAREST)) >= 250
    count = int(skin.sum())
    record["skin_pixels"] = count
    if count < 64:
        return {**record, "status": "insufficient_visible_skin", "warnings": [], "baseline_penalty": 0.}
    pixels = values[skin]
    chroma = np.median(pixels/np.maximum(pixels.sum(1, keepdims=True), 1e-6), axis=0)
    chroma /= max(float(chroma.sum()), 1e-6)
    luminance = values @ np.array([.2126, .7152, .0722], dtype=np.float32)
    halves = [luminance[:, start:stop][skin[:, start:stop]] for start, stop in ((0, 64), (64, 128))]
    ratio = None
    if all(len(half) >= 32 for half in halves):
        medians = [float(np.median(half)) for half in halves]
        ratio = max(medians)/max(min(medians), 1e-5)
    return {
        **record, "status": "observed", "chromaticity": chroma.tolist(),
        "skin_luma_median": float(np.median(luminance[skin])),
        "left_right_luma_ratio": ratio,
        "clipped_fraction": float(((pixels.max(1) >= 252/255) | (pixels.max(1) <= 3/255)).mean()),
        "warnings": [], "baseline_penalty": 0.,
    }


def assess_lighting(records):
    """Compare only within a fixed subject hypothesis; never change pixel colour."""
    import numpy as np

    references = {}
    scopes = {record["subject_scope"] for record in records.values()}
    for scope in sorted(scopes):
        eligible = [
            record["lighting"]["chromaticity"] for record in records.values()
            if record["subject_scope"] == scope and record["status"] == "ready"
            and record["view"]["category"] == "front" and not record["quality_warnings"]
            and record["lighting"]["status"] == "observed"
            and record["lighting"]["clipped_fraction"] < .05
            and (record["lighting"]["left_right_luma_ratio"] or 1) < 2.5
        ]
        if len(eligible) < 3:
            continue
        samples = np.asarray(eligible)
        center = np.median(samples, axis=0)
        center /= center.sum()
        distances = np.linalg.norm(samples - center, axis=1)
        median = float(np.median(distances))
        deviation = float(np.median(np.abs(distances - median)))
        threshold = max(.035, median + 3*1.4826*deviation)
        references[scope] = {"chromaticity": center.tolist(), "outlier_distance": threshold,
                             "front_references": len(eligible)}
    for record in records.values():
        light = record["lighting"]
        for field in ("subject_chromaticity_distance", "outlier_distance", "reference_status"):
            light.pop(field, None)
        if light["status"] != "observed":
            continue
        warnings, penalty = [], 0.
        reference = references.get(record["subject_scope"])
        if reference is not None:
            distance = float(np.linalg.norm(np.asarray(light["chromaticity"]) - reference["chromaticity"]))
            light.update(subject_chromaticity_distance=distance, outlier_distance=reference["outlier_distance"],
                         reference_status="relative_subject_hypothesis")
            penalty += .35*min(1., distance/reference["outlier_distance"])
            if distance > reference["outlier_distance"]:
                warnings.append("relative_subject_colour_outlier")
        else:
            light["reference_status"] = "insufficient_clean_frontal_references"
        ratio = light["left_right_luma_ratio"] or 1
        penalty += .35*min(1., max(0., ratio - 1)/2)
        penalty += .3*min(1., light["clipped_fraction"]/.08)
        if ratio > 2.5:
            warnings.append("strong_left_right_skin_luma_difference")
        if light["clipped_fraction"] > .05:
            warnings.append("clipped_visible_skin")
        light.update(warnings=warnings, baseline_penalty=round(penalty, 5))
    return {"method": "robust within-subject frontal chromaticity; heuristic, not calibrated light or albedo",
            "identity_confirmed": False, "references": references}

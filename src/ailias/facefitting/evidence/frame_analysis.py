"""Analyze per-image face landmarks, pose and quality without decoding video."""

import math

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


EXPRESSION_PREFIXES = ("mouth", "brow", "cheek", "noseSneer", "eyeWide", "eyeSquint")

def analyze_face_frame(rgb, detection, policy):
    """Keep source-pixel coordinates; pose/depth/expression remain model predictions."""
    import cv2

    points = np.asarray(detection["faces"][0], dtype=float) if len(detection["faces"]) == 1 else None
    if points is None or points.shape != (478, 3) or not np.isfinite(points).all():
        return {"eligible_identity": False, "reasons": ["invalid_or_multiple_faces"]}
    height, width = rgb.shape[:2]
    xy = points[:, :2]*[width, height]
    inside = ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all(1)
    span = float(np.linalg.norm(xy[33]-xy[263]))
    low, high = xy[:468].min(0), xy[:468].max(0)
    x0, y0 = np.clip(np.floor(low).astype(int), 0, [width-1, height-1])
    x1, y1 = np.clip(np.ceil(high).astype(int), 1, [width, height])
    if x1-x0 < 3 or y1-y0 < 3:
        return {"eligible_identity": False, "reasons": ["invalid_face_bounds"]}
    crop = Image.fromarray(rgb).crop((x0, y0, x1, y1)).convert("L")
    crop.thumbnail((256, 256))
    gray = np.asarray(crop)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blends = detection["blendshapes"][0] if detection["blendshapes"] else {}
    expression = max((value for key, value in blends.items()
                      if key.startswith(EXPRESSION_PREFIXES)), default=1.)
    jaw = float(blends.get("jawOpen", 1.))
    blink = max(blends.get("eyeBlinkLeft", 1.), blends.get("eyeBlinkRight", 1.))
    transform = np.asarray(detection["transforms"][0], dtype=float) if detection["transforms"] else None
    if (transform is None or transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=.02)):
        return {"eligible_identity": False, "reasons": ["invalid_pose_transform"]}
    pitch, yaw, roll = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    category = "front" if abs(yaw) < 15 else ("negative" if yaw < 0 else "positive") + (
        "_oblique" if abs(yaw) < 50 else "_profile")
    reasons = []
    if span < policy.minimum_eye_span_pixels:
        reasons.append("face_too_small")
    if sharpness < policy.minimum_laplacian_variance:
        reasons.append("blur_or_low_detail")
    if not inside[:468].all():
        reasons.append("face_clipped")
    if not blends:
        reasons.append("missing_expression_evidence")
    if expression > policy.neutral_expression_max or jaw > policy.jaw_open_max:
        reasons.append("non_neutral_expression")
    if blink > policy.blink_max:
        reasons.append("blink")
    if abs(pitch) > 25 or abs(yaw) > 70 or abs(roll) > 25:
        reasons.append("extreme_pose")
    dark, bright = float((gray < 12).mean()), float((gray > 245).mean())
    if dark > .3 or bright > .2:
        reasons.append("poor_exposure")
    return {
        "source_size": [width, height], "source_xy": xy.tolist(),
        "on_head": inside.tolist(), "eye_span_source": span,
        "landmark_depth_is_measured": False, "per_point_visibility_available": False,
        "blendshapes": blends, "facial_transform": transform.tolist(),
        "pose": {"yaw_deg": float(yaw), "pitch_deg": float(pitch),
                 "roll_deg": float(roll), "category": category, "calibrated": False},
        "face_box": [int(x0), int(y0), int(x1), int(y1)],
        "quality": {"laplacian_variance": sharpness, "dark_fraction": dark,
                    "bright_fraction": bright, "expression_max": float(expression),
                    "jaw_open": jaw, "blink_max": float(blink)},
        "eligible_identity": not reasons, "reasons": reasons,
        "quality_score": math.log1p(sharpness)-4*expression-4*jaw-2*blink,
    }

"""Approximate view categorization, not calibrated cameras or capture attestations."""

import math

# MediaPipe v0.10.21 canonical_face_model.obj: eye-corner means, nose and mouth corners.
# Convert its Y-up/Z-out coordinates to image-right/Y-down/Z-back. See research.md.
POSE_POINTS = (
    (-3.1511455, -2.624618, -3.465663), (3.1511455, -2.624618, -3.465663),
    (0, 1.126865, -7.475604), (-2.456206, 4.342621, -4.283884),
    (2.456206, 4.342621, -4.283884),
)

VIEW_WEIGHTS = {
    "front": 0.50, "left_oblique": 0.12, "right_oblique": 0.12,
    "left_profile": 0.07, "right_profile": 0.07,
    "up": 0.04, "down": 0.04, "unknown": 0.04,
}


def view_bin(yaw, pitch):
    if abs(pitch) > 20 and abs(yaw) < 35:
        return "up" if pitch > 0 else "down"
    if abs(yaw) <= 20:
        return "front"
    side = "left" if yaw < 0 else "right"
    return side + ("_oblique" if abs(yaw) < 55 else "_profile")


def estimate_view(face, metadata):
    import cv2
    import numpy as np

    unknown = {"category": "unknown", "confidence": 0.0, "calibrated": False,
               "method": "five-point-canonical-sqpnp-focal-sensitivity-v1"}
    landmarks = np.asarray(face.get("landmarks_5", []), dtype=np.float64)
    size = metadata.get("oriented_size")
    if (landmarks.shape != (5, 2) or not np.isfinite(landmarks).all()
            or not size or len(size) != 2 or min(size) <= 0):
        return {**unknown, "reason": "missing_or_invalid_landmarks_or_source_size"}
    eye_distance = float(np.linalg.norm(landmarks[0] - landmarks[1]))
    if eye_distance < 4:
        return {**unknown, "reason": "insufficient_visible_eye_baseline"}
    points = np.asarray(POSE_POINTS, dtype=np.float64)
    hypotheses = []
    for factor in (0.8, 1.2, 1.8):
        focal = factor * max(size)
        intrinsics = np.array([[focal, 0, size[0]/2], [0, focal, size[1]/2], [0, 0, 1.]])
        try:
            success, rotation, translation = cv2.solvePnP(
                points, landmarks, intrinsics, None, flags=cv2.SOLVEPNP_SQPNP)
            if not success:
                continue
            matrix = cv2.Rodrigues(rotation)[0]
            camera_points = points @ matrix.T + translation.reshape(3)
            if not np.isfinite(camera_points).all() or camera_points[:, 2].min() <= 0:
                continue
            predicted = cv2.projectPoints(points, rotation, translation, intrinsics, None)[0].reshape(5, 2)
            error = float(np.linalg.norm(predicted - landmarks, axis=1).mean()/eye_distance)
            direction = matrix.T @ -translation.reshape(3)
            yaw = math.degrees(math.atan2(direction[0], -direction[2]))
            pitch = math.degrees(math.atan2(-direction[1], math.hypot(direction[0], direction[2])))
            if abs(yaw) < 100 and abs(pitch) < 75 and math.isfinite(error):
                hypotheses.append((yaw, pitch, error))
        except cv2.error:
            continue
    if len(hypotheses) < 2:
        return {**unknown, "reason": "ambiguous_or_unusable_pose_fit"}
    values = np.asarray(hypotheses)
    yaw, pitch, error = np.median(values, axis=0)
    spread = float(max(np.ptp(values[:, 0]), np.ptp(values[:, 1])))
    confidence = max(0., 1 - error/0.18) * max(0., 1 - spread/35)
    return {
        **unknown, "category": view_bin(yaw, pitch) if confidence >= 0.25 else "unknown",
        "yaw_deg": round(float(yaw), 2), "pitch_deg": round(float(pitch), 2),
        "confidence": round(confidence, 4), "normalized_reprojection_error": round(float(error), 4),
        "focal_sensitivity_deg": round(spread, 2),
        "laterality": "camera on canonical -X/+X; not anatomical side certification",
    }

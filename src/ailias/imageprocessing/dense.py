"""Predict dense-face landmarks and map aligned points back to source pixels.

MediaPipe outputs are image evidence, not measured depth or calibrated visibility.
"""

import numpy as np
from PIL import Image


class DenseDetector:
    def __init__(self, model, *, device="cpu"):
        """Create MediaPipe's face-detection/landmark bundle with an explicit CPU or GPU delegate.
        Preserve legacy CPU callers; an unavailable requested delegate fails without a CPU retry.
        """
        if device not in {"cpu", "gpu"}:
            raise ValueError("MediaPipe delegate must be cpu or gpu")
        import mediapipe as mp

        self.mp, self.device = mp, device
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model), delegate=(
                        mp.tasks.BaseOptions.Delegate.GPU if device == "gpu"
                        else mp.tasks.BaseOptions.Delegate.CPU)),
                num_faces=2, output_face_blendshapes=True,
                output_facial_transformation_matrixes=True))

    def detect(self, rgb):
        """Pass contiguous RGB pixels through the configured MediaPipe neural-model bundle.
        Return predicted points, expression scores and transforms without changing the selected delegate.
        """
        result = self.detector.detect(self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return {
            "faces": [[[p.x, p.y, p.z] for p in face] for face in result.face_landmarks],
            "blendshapes": [{p.category_name: p.score for p in face}
                            for face in result.face_blendshapes],
            "transforms": [matrix.tolist() for matrix in result.facial_transformation_matrixes],
        }

    def close(self):
        """Close the owned MediaPipe task and its configured inference resources.
        Leave other detectors and external runtime processes untouched.
        """
        self.detector.close()


def observe(store, entry, source_size, detector):
    analysis = entry.get("analysis", {})
    artifact = analysis.get("aligned_cutout")
    if (not artifact or not store.verify(artifact)
            or analysis.get("alignment", {}).get("status") != "ready"):
        raise ValueError("Require verified aligned head and reversible source coordinates")
    with Image.open(store.artifact_path(artifact)) as image:
        rgba = np.asarray(image.convert("RGBA"))
    result = detector.detect(rgba[:, :, :3])
    record = {
        "face_id": entry["face_id"], "source_id": entry["source_id"],
        "aligned_cutout": artifact, "alignment": analysis["alignment"],
        "source_size": source_size, "size": [rgba.shape[1], rgba.shape[0]],
        "device": f"MediaPipe {getattr(detector, 'device', 'cpu').upper()} (explicit); existing CUDA cleansing is unchanged",
        "landmark_depth_is_measured": False, "per_point_visibility_available": False,
    }
    if len(result["faces"]) != 1:
        return {**record, "status": "review", "reason": "dense_face_count_not_one"}
    landmarks = np.asarray(result["faces"][0], dtype=float)
    if landmarks.shape != (478, 3) or not np.isfinite(landmarks).all():
        return {**record, "status": "review", "reason": "invalid_dense_landmarks"}
    aligned = landmarks[:, :2] * record["size"]
    transform = np.asarray(analysis["alignment"]["aligned_to_source"], dtype=float)
    inverse = np.asarray(analysis["alignment"]["source_to_aligned"], dtype=float)
    if (transform.shape != (3, 3) or not np.isfinite(transform).all()
            or not np.allclose(inverse @ transform, np.eye(3), atol=1e-6)):
        raise ValueError("Invalid aligned-to-source transformation")
    source = (np.c_[aligned, np.ones(len(aligned))] @ transform.T)[:, :2]
    inside = ((aligned >= 0) & (aligned < record["size"])).all(1)
    pixels = np.clip(np.floor(aligned).astype(int), 0, np.asarray(record["size"]) - 1)
    on_head = inside & (rgba[pixels[:, 1], pixels[:, 0], 3] >= 240)
    eye_spacing = float(np.linalg.norm(aligned[33] - aligned[263]))
    if eye_spacing < 12 or int(on_head[:468].sum()) < 150:
        return {**record, "status": "review", "reason": "insufficient_dense_head_support"}
    return {
        **record, "status": "ready", "landmarks_normalized_predicted": landmarks.tolist(),
        "aligned_xy": aligned.tolist(), "source_xy": source.tolist(),
        "on_head": on_head.tolist(),
        "eye_span_source": float(np.linalg.norm(source[33] - source[263])),
        "eye_span_definition": "outer eye-corner distance; not physical interpupillary distance",
        "blendshapes": result["blendshapes"][0] if result["blendshapes"] else {},
        "facial_transform": result["transforms"][0] if result["transforms"] else None,
    }

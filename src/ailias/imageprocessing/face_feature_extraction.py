"""Extract facial landmarks, measurements and regional appearance from prepared images.

Preparation and model-free image operations belong to the separate images module.
"""

from collections.abc import Iterable, Iterator
from dataclasses import replace
import math
from pathlib import Path
import platform

from PIL import Image

from ailias.imageprocessing.face_features import (
    FaceFeatures, FaceFeaturesBiseNet as FaceFeaturesBiseNet,
)
from ailias.imageprocessing.settings import (
    MEDIAPIPE_BASE_FACE_POINT_COUNT, MEDIAPIPE_LANDMARK_COUNT, ProcessingPolicy, load_policy,
)


NEUTRAL_BACKGROUND = (127, 127, 127, 255)
# Allow small numerical drift in a unit orthogonal rotation, not extra head-angle deviation.
ROTATION_VALIDATION_TOLERANCE = 0.02


class FaceFeatureExtractor:
    """Extract evidence from existing pixels and region labels without preparing them again.
    Reuse one lazy MediaPipe task across a collection and release it on context exit.
    """

    def __init__(
        self, *, mediapipe_model: str | Path | None = None, device: str = "cuda",
        quality_policy: ProcessingPolicy | None = None,
    ):
        """Accept an existing MediaPipe bundle and validated review policy without loading weights.
        Default to CUDA mode; choose the platform-specific delegate only when inference is requested.
        """
        if device not in {"cpu", "cuda"}:
            raise ValueError("Feature extraction device must be cpu or cuda")
        self.device = device
        self.landmarker_device = "cpu" if device == "cpu" or platform.system() == "Windows" else "gpu"
        self.quality_policy = quality_policy if quality_policy is not None else load_policy()
        self._model_path = Path(mediapipe_model) if mediapipe_model is not None else None
        self._landmarker = None
        self._closed = False

    def __enter__(self):
        """Keep the extractor and its model alive for the enclosing collection.
        Reject reuse after close rather than silently constructing another runtime.
        """
        if self._closed:
            raise RuntimeError("Face feature extractor is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release owned inference resources when extraction finishes or fails.
        Allow the original exception to propagate without a fallback attempt.
        """
        self.close()

    def close(self) -> None:
        """Release the owned MediaPipe task without touching preparation networks.
        Closing is idempotent and never shuts down an external runtime process.
        """
        self._closed = True
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            landmarker.close()

    def detect_face_landmarks(self, image: Image.Image) -> dict:
        """Use MediaPipe's face-detection/landmark bundle on neutral-grey RGB for points and expressions.
        Windows CUDA mode uses CPU because MediaPipe's Python GPU delegate is unsupported on Windows.
        Other CUDA platforms request GPU; explicit CPU mode stays on CPU, without retry fallback.
        """
        import numpy as np

        if self._closed:
            raise RuntimeError("Face feature extractor is closed")
        if self._landmarker is None:
            if self._model_path is None:
                raise ValueError("Supply the existing mediapipe model path")
            if not self._model_path.is_file():
                raise FileNotFoundError(self._model_path)
            from ailias.imageprocessing.dense import DenseDetector

            self._landmarker = DenseDetector(self._model_path, device=self.landmarker_device)
        composite = Image.new("RGBA", image.size, NEUTRAL_BACKGROUND)
        composite.alpha_composite(image.convert("RGBA"))
        return self._landmarker.detect(np.asarray(composite.convert("RGB")))

    def extract_features(self, prepared_image: dict) -> FaceFeatures:
        """Extract MediaPipe predictions and original-pixel appearance using existing region labels.
        Return a FaceFeatures instance without matting, parsing, colour correction or file writes.
        """
        if self._closed:
            raise RuntimeError("Face feature extractor is closed")
        from ailias.imageprocessing.appearance_features import extract_bisenet_features

        original_rgb, head, regions = self._validate_prepared_image(prepared_image)
        result = self._create_feature_result(prepared_image)
        if head.getchannel("A").getbbox() is None:
            result.quality["reasons"].append("no_head_foreground")
            return result
        result = replace(result, bisenet=extract_bisenet_features(original_rgb, regions))
        detection = self.detect_face_landmarks(head)
        result.execution["mediapipe_executed"] = True
        return replace(
            result, **self._extract_face_evidence(original_rgb, head, regions, detection, result.bisenet))

    @staticmethod
    def _validate_prepared_image(prepared_image: dict) -> tuple[Image.Image, Image.Image, Image.Image]:
        """Require a prepared RGBA head, original colour pixels and grayscale semantic labels.
        Reject mismatched grids before inference so landmarks and colour regions cannot silently drift.
        """
        head = prepared_image.get("image")
        original_rgb = prepared_image.get("original_image")
        regions = prepared_image.get("regions")
        if not isinstance(head, Image.Image) or head.mode != "RGBA":
            raise ValueError("Prepared image must contain an RGBA head image")
        if not isinstance(original_rgb, Image.Image):
            raise ValueError("Prepared image must contain original colour pixels")
        if not isinstance(regions, Image.Image) or regions.mode != "L":
            raise ValueError("Prepared image must contain grayscale semantic region labels")
        if original_rgb.size != head.size or regions.size != head.size:
            raise ValueError("Prepared image, original pixels and region labels must share one pixel grid")
        return original_rgb.convert("RGB"), head, regions

    def _create_feature_result(self, prepared_image: dict) -> FaceFeatures:
        """Create feature-only output with source identity and explicit inference provenance.
        Leave image preparation products owned by the caller instead of copying them into feature evidence.
        """
        return FaceFeatures(
            source=prepared_image.get("source"), frame_index=prepared_image.get("frame_index"),
            timestamp_seconds=prepared_image.get("timestamp_seconds"),
            execution={
                "requested_device": self.device, "mediapipe_device": self.landmarker_device,
                "mediapipe_executed": False,
            },
        )

    def _extract_face_evidence(
        self, original_rgb: Image.Image, head: Image.Image, regions: Image.Image, detection: dict,
        bisenet: FaceFeaturesBiseNet,
    ) -> dict:
        """Combine MediaPipe points and existing BiSeNet labels into original-pixel feature evidence.
        Keep invalid-face review outcomes separate from coordinate mapping and quality assessment.
        """
        import numpy as np
        from ailias.imageprocessing.appearance_features import photo_features

        if len(detection["faces"]) != 1:
            reason = "no_face_detected" if not detection["faces"] else "multiple_faces_detected"
            return {"quality": {"reasons": [reason]}}
        landmarks = self._map_face_landmarks(detection["faces"][0], head)
        if "review_reason" in landmarks:
            return {"quality": {"reasons": [landmarks["review_reason"]]}}
        blendshapes = detection["blendshapes"][0] if detection["blendshapes"] else {}
        transform = detection["transforms"][0] if detection["transforms"] else None
        quality = self._assess_face_quality(original_rgb, landmarks, blendshapes, transform)
        return {
            "status": "review" if quality["reasons"] else "ready", "quality": quality,
            "landmarks": {
                "normalized_predicted": landmarks["normalized_predicted"].tolist(),
                "source_xy": landmarks["source_xy"].tolist(), "on_head": landmarks["on_head"].tolist(),
            },
            "blendshapes": blendshapes, "facial_transform": transform,
            "features": photo_features(np.asarray(original_rgb), landmarks["source_xy"], regions,
                                       landmarks["on_head"], region_features=bisenet),
        }

    def _map_face_landmarks(self, normalized_points: list, head: Image.Image) -> dict:
        """Map MediaPipe's 478 normalized predictions onto the unchanged source-image grid.
        Derive face bounds and alpha support without treating predicted depth as measured geometry.
        """
        import numpy as np

        normalized = np.asarray(normalized_points, dtype=float)
        if normalized.shape != (MEDIAPIPE_LANDMARK_COUNT, 3) or not np.isfinite(normalized).all():
            return {"review_reason": "invalid_dense_landmarks"}
        source_xy = normalized[:, :2] * head.size
        inside_image = ((source_xy >= 0) & (source_xy < head.size)).all(1)
        sample_pixels = np.clip(np.floor(source_xy).astype(int), 0, np.asarray(head.size) - 1)
        alpha = np.asarray(head.getchannel("A"))
        on_head = inside_image & (
            alpha[sample_pixels[:, 1], sample_pixels[:, 0]] >= self.quality_policy.min_landmark_alpha)
        face_points = source_xy[:MEDIAPIPE_BASE_FACE_POINT_COUNT]
        minimum_xy, maximum_xy = face_points.min(0), face_points.max(0)
        left, top = np.maximum(0, minimum_xy)
        right, bottom = np.minimum(head.size, maximum_xy)
        if right <= left or bottom <= top:
            return {"review_reason": "face_outside_image"}
        return {
            "normalized_predicted": normalized, "source_xy": source_xy,
            "inside_image": inside_image, "on_head": on_head,
            "face_box": [left, top, right - left, bottom - top],
        }

    def _assess_face_quality(
        self, original_rgb: Image.Image, landmarks: dict, blendshapes: dict, transform,
    ) -> dict:
        """Assess image detail, clipping, head support, expression and predicted pose independently.
        Return review reasons without discarding usable features or claiming calibrated confidence.
        """
        from ailias.imageprocessing.features import face_quality

        quality = face_quality(original_rgb, landmarks["face_box"], self.quality_policy)
        if not landmarks["inside_image"][:MEDIAPIPE_BASE_FACE_POINT_COUNT].all():
            quality["reasons"].append("face_clipped")
        supported_points = int(landmarks["on_head"][:MEDIAPIPE_BASE_FACE_POINT_COUNT].sum())
        if supported_points < self.quality_policy.min_supported_landmarks:
            quality["reasons"].append("insufficient_head_support")
        expression = max((value for name, value in blendshapes.items()
                          if name.startswith(("mouth", "brow", "cheek", "noseSneer", "eyeWide", "eyeSquint"))
                          or name == "jawOpen"), default=0.0)
        if expression > self.quality_policy.max_neutral_expression:
            quality["reasons"].append("non_neutral_expression")
        pose, pose_warning = self._assess_face_pose(transform)
        if pose is not None:
            quality["pose"] = pose
        if pose_warning is not None:
            quality["reasons"].append(pose_warning)
        return quality

    def _assess_face_pose(self, transform) -> tuple[dict | None, str | None]:
        """Validate the predicted rotation and derive uncalibrated pitch, yaw and roll.
        Flag unavailable or extreme pose without fitting a camera or altering image pixels.
        """
        import numpy as np

        matrix = np.asarray(transform, dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3),
                                   atol=ROTATION_VALIDATION_TOLERANCE)
                or not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1.0,
                                    abs_tol=ROTATION_VALIDATION_TOLERANCE)):
            return None, "missing_or_invalid_pose"
        yaw = math.degrees(math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0])))
        pitch = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
        roll = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        pose = {"pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll, "calibrated": False}
        extreme = (abs(yaw) > self.quality_policy.max_yaw_degrees
                   or max(abs(pitch), abs(roll)) > self.quality_policy.max_pitch_roll_degrees)
        return pose, "extreme_pose" if extreme else None

    def extract_features_batch(self, prepared_images: Iterable[dict]) -> Iterator[FaceFeatures]:
        """Stream FaceFeatures instances while reusing one landmarker and existing prepared pixels.
        Preserve preparation failures and per-image extraction errors as class instances, not dictionaries.
        """
        for prepared in prepared_images:
            if prepared.get("status") == "error":
                yield self._create_error_result(prepared)
                continue
            try:
                yield self.extract_features(prepared)
            except (OSError, ValueError, RuntimeError) as error:
                yield self._create_error_result(prepared, error)

    def _create_error_result(self, prepared_image: dict, error: Exception | None = None) -> FaceFeatures:
        """Represent a preparation or extraction failure with the same FaceFeatures output contract.
        Preserve source identity without claiming successful model execution or fabricated observations.
        """
        return replace(
            self._create_feature_result(prepared_image), status="error", execution={},
            error_type=type(error).__name__ if error is not None else prepared_image.get("error_type"),
            reason=str(error) if error is not None else prepared_image.get("reason"),
        )

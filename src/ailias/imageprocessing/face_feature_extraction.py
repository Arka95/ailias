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
    FaceExtractionResult,
    FaceFeatures,
    FaceFeaturesBiseNet as FaceFeaturesBiseNet,
    FaceQuality,
)
from ailias.imageprocessing.settings import (
    MEDIAPIPE_BASE_FACE_POINT_COUNT,
    MEDIAPIPE_LANDMARK_COUNT,
    FaceQualityPolicy,
    load_face_quality_policy,
)


NEUTRAL_BACKGROUND = (127, 127, 127, 255)
# Allow numerical drift after removing uniform scale, not non-rigid shear or reflected coordinates.
ROTATION_VALIDATION_TOLERANCE = 0.02


class FaceFeatureExtractor:
    """Extract one known person's facial evidence from existing pixels and region labels.
    Reuse one lazy MediaPipe task without identifying people or selecting collection members.
    """

    def __init__(
        self,
        *,
        mediapipe_model: str | Path | None = None,
        device: str = "cuda",
        quality_policy: FaceQualityPolicy | None = None,
    ):
        """Accept a local MediaPipe bundle and quality policy without loading weights or selecting images.
        Default to CUDA mode; choose the platform-specific delegate only when inference is requested.
        """
        if device not in {"cpu", "cuda"}:
            raise ValueError("Feature extraction device must be cpu or cuda")
        self.device = device
        self.quality_policy = (
            quality_policy if quality_policy is not None else load_face_quality_policy()
        )
        self.landmarker_device = (
            "cpu" if device == "cpu" or platform.system() == "Windows" else "gpu"
        )
        self._model_path = (
            Path(mediapipe_model) if mediapipe_model is not None else None
        )
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

            self._landmarker = DenseDetector(
                self._model_path, device=self.landmarker_device
            )
        composite = Image.new("RGBA", image.size, NEUTRAL_BACKGROUND)
        composite.alpha_composite(image.convert("RGBA"))
        return self._landmarker.detect(np.asarray(composite.convert("RGB")))

    def extract_features(self, prepared_image: dict) -> FaceFeatures:
        """Extract MediaPipe predictions and original-pixel appearance using existing region labels.
        Return only facial data; batch results additionally carry provenance and failure reasons.
        """
        return self.extract_feature_result(prepared_image).features or FaceFeatures(
            source_size=prepared_image["image"].size
        )

    def extract_feature_result(self, prepared_image: dict) -> FaceExtractionResult:
        """Extract facial data with source/frame metadata and explicit per-view failure reasons.
        Retain partial evidence; let runtime exceptions propagate rather than hiding them in a batch.
        """
        if self._closed:
            raise RuntimeError("Face feature extractor is closed")
        from ailias.imageprocessing.appearance_features import extract_bisenet_features

        original_rgb, head, regions = self._validate_prepared_image(prepared_image)
        result = self._create_extraction_result(prepared_image)
        if head.getchannel("A").getbbox() is None:
            return replace(result, error_type="ValueError", reason="no_head_foreground")
        bisenet = extract_bisenet_features(original_rgb, regions)
        result = replace(
            result, features=FaceFeatures(bisenet=bisenet, source_size=head.size)
        )
        detection = self.detect_face_landmarks(head)
        result.execution["mediapipe_executed"] = True
        if len(detection["faces"]) != 1:
            reason = (
                "no_face_detected"
                if not detection["faces"]
                else "ambiguous_face_detections"
            )
            return replace(result, error_type="ValueError", reason=reason)
        try:
            features = self._extract_face_measurements(
                original_rgb, head, regions, detection, bisenet
            )
        except ValueError as error:
            return replace(result, error_type="ValueError", reason=str(error))
        return replace(result, features=features)

    @staticmethod
    def _validate_prepared_image(
        prepared_image: dict,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
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
            raise ValueError(
                "Prepared image must contain grayscale semantic region labels"
            )
        if original_rgb.size != head.size or regions.size != head.size:
            raise ValueError(
                "Prepared image, original pixels and region labels must share one pixel grid"
            )
        return original_rgb.convert("RGB"), head, regions

    def _create_extraction_result(self, prepared_image: dict) -> FaceExtractionResult:
        """Keep source/frame/backend metadata alongside facial measurements rather than inside them.
        Retain supplied pixels by reference for grouping and review; never reload or alter the source.
        """
        return FaceExtractionResult(
            image=prepared_image.get("original_image"),
            head_image=prepared_image.get("image"),
            source=prepared_image.get("source"),
            frame_index=prepared_image.get("frame_index"),
            timestamp_seconds=prepared_image.get("timestamp_seconds"),
            execution={
                "requested_device": self.device,
                "mediapipe_device": self.landmarker_device,
                "mediapipe_executed": False,
            },
        )

    def _extract_face_measurements(
        self,
        original_rgb: Image.Image,
        head: Image.Image,
        regions: Image.Image,
        detection: dict,
        bisenet: FaceFeaturesBiseNet,
    ) -> FaceFeatures:
        """Combine MediaPipe points and existing BiSeNet labels into original-pixel feature evidence.
        Calculate orientation and structured quality once without selecting or rejecting collection members.
        """
        import numpy as np
        from ailias.imageprocessing.appearance_features import photo_features
        from ailias.imageprocessing.hair_evidence import facial_hair_evidence

        landmarks = self._map_face_landmarks(detection["faces"][0], head)
        blendshapes = detection["blendshapes"][0] if detection["blendshapes"] else {}
        transform = detection["transforms"][0] if detection["transforms"] else None
        features = photo_features(
            np.asarray(original_rgb),
            landmarks["source_xy"],
            regions,
            landmarks["on_head"],
            region_features=bisenet,
        )
        if bisenet.skin.status == "observed":
            features["facial_hair_appearance"] = facial_hair_evidence(
                np.asarray(original_rgb),
                landmarks["source_xy"],
                np.asarray(regions),
                bisenet.skin.median_linear_rgb,
            )
        return FaceFeatures(
            source_size=head.size,
            bisenet=bisenet,
            landmarks={
                "normalized_predicted": landmarks["normalized_predicted"].tolist(),
                "source_xy": landmarks["source_xy"].tolist(),
                "on_head": landmarks["on_head"].tolist(),
            },
            blendshapes=blendshapes,
            facial_transform=transform,
            orientation=determine_face_orientation(transform),
            quality=self._assess_face_quality(
                original_rgb, head, landmarks["source_xy"], blendshapes
            ),
            features=features,
        )

    def _map_face_landmarks(self, normalized_points: list, head: Image.Image) -> dict:
        """Map MediaPipe's 478 normalized predictions onto the unchanged source-image grid.
        Record visible-alpha support without applying quality thresholds, pose limits or selection policy.
        """
        import numpy as np

        normalized = np.asarray(normalized_points, dtype=float)
        if (
            normalized.shape != (MEDIAPIPE_LANDMARK_COUNT, 3)
            or not np.isfinite(normalized).all()
        ):
            raise ValueError("invalid_dense_landmarks")
        source_xy = normalized[:, :2] * head.size
        inside_image = ((source_xy >= 0) & (source_xy < head.size)).all(1)
        sample_pixels = np.clip(
            np.floor(source_xy).astype(int), 0, np.asarray(head.size) - 1
        )
        alpha = np.asarray(head.getchannel("A"))
        on_head = inside_image & (alpha[sample_pixels[:, 1], sample_pixels[:, 0]] > 0)
        return {
            "normalized_predicted": normalized,
            "source_xy": source_xy,
            "on_head": on_head,
        }

    def _assess_face_quality(
        self, original_rgb: Image.Image, head: Image.Image, points, blendshapes: dict
    ) -> FaceQuality:
        """Measure detail, exposure, clipping, alpha support and expression during feature extraction.
        Retain measured counts/scores and warnings without penalizing angle or deciding selection.
        """
        import numpy as np
        from ailias.imageprocessing.features import face_quality

        base = points[:MEDIAPIPE_BASE_FACE_POINT_COUNT]
        inside = ((base >= 0) & (base < original_rgb.size)).all(1)
        samples = np.clip(np.floor(base).astype(int), 0, np.asarray(head.size) - 1)
        alpha = np.asarray(head.getchannel("A"))
        supported = inside & (
            alpha[samples[:, 1], samples[:, 0]]
            >= self.quality_policy.min_landmark_alpha
        )
        expression = float(
            max(
                (
                    value
                    for name, value in blendshapes.items()
                    if name.startswith(
                        ("mouth", "brow", "cheek", "noseSneer", "eyeWide", "eyeSquint")
                    )
                    or name == "jawOpen"
                ),
                default=0.0,
            )
        )
        low, high = (
            np.maximum(0, base.min(0)),
            np.minimum(original_rgb.size, base.max(0)),
        )
        if np.any(high <= low):
            metrics = {"reasons": ["face_outside_image_or_degenerate_landmarks"]}
        else:
            box = [*low.tolist(), *(high - low).tolist()]
            metrics = face_quality(original_rgb, box, self.quality_policy)
        reasons = metrics["reasons"]
        if not inside.all():
            reasons.append("face_clipped")
        if int(supported.sum()) < self.quality_policy.min_supported_landmarks:
            reasons.append("insufficient_head_support")
        if expression > self.quality_policy.max_neutral_expression:
            reasons.append("non_neutral_expression")
        return FaceQuality(
            **metrics,
            supported_landmarks=int(supported.sum()),
            clipped_landmarks=int((~inside).sum()),
            max_expression_score=expression,
        )

    def extract_features_batch(
        self, prepared_images: Iterable[dict]
    ) -> Iterator[FaceExtractionResult]:
        """Stream facial data together with source metadata and per-image extraction failures.
        Reuse one landmarker and keep original pixels available for subsequent grouping and review copies.
        """
        for prepared in prepared_images:
            if prepared.get("status") == "error":
                yield self._create_error_result(prepared)
                continue
            try:
                yield self.extract_feature_result(prepared)
            except (OSError, ValueError, RuntimeError) as error:
                yield self._create_error_result(prepared, error)

    def _create_error_result(
        self, prepared_image: dict, error: Exception | None = None
    ) -> FaceExtractionResult:
        """Represent a preparation or extraction failure outside the facial-data class.
        Preserve source identity without claiming successful model execution or fabricated observations.
        """
        return replace(
            self._create_extraction_result(prepared_image),
            execution={},
            error_type=type(error).__name__
            if error is not None
            else prepared_image.get("error_type"),
            reason=str(error) if error is not None else prepared_image.get("reason"),
        )


def determine_face_orientation(transform) -> dict:
    """Derive approximate model-space Euler angles from MediaPipe's raw face transform.
    Remove uniform scale before validating rotation; unavailable orientation is a diagnostic, not an error.
    """
    import numpy as np

    unknown = {"calibrated": False, "reason": "missing_or_invalid_pose"}
    try:
        matrix = np.asarray(transform, dtype=float)
    except (TypeError, ValueError):
        return unknown
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return unknown
    linear = matrix[:3, :3]
    scale = float(np.linalg.norm(linear[:, 0]))
    if scale <= np.finfo(float).eps:
        return unknown
    # MediaPipe's canonical-to-runtime transform can contain uniform scale as well as rotation.
    rotation = linear / scale
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=ROTATION_VALIDATION_TOLERANCE)
        or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=ROTATION_VALIDATION_TOLERANCE
        )
        or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=ROTATION_VALIDATION_TOLERANCE
        )
    ):
        return unknown
    yaw = math.degrees(
        math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    )
    pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return {"pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll, "calibrated": False}

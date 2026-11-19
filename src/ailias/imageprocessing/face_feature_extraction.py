"""Extract facial landmarks, measurements and regional appearance from prepared images.

Preparation and model-free image operations belong to the separate images module.
"""

from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
import platform

from PIL import Image

from ailias.imageprocessing.face_features import (
    FaceExtractionResult, FaceFeatures, FaceFeaturesBiseNet as FaceFeaturesBiseNet,
)
from ailias.imageprocessing.settings import (
    MEDIAPIPE_LANDMARK_COUNT,
)


NEUTRAL_BACKGROUND = (127, 127, 127, 255)


class FaceFeatureExtractor:
    """Extract one known person's facial evidence from existing pixels and region labels.
    Reuse one lazy MediaPipe task without identifying people or selecting collection members.
    """

    def __init__(
        self, *, mediapipe_model: str | Path | None = None, device: str = "cuda",
    ):
        """Accept an existing MediaPipe bundle without loading weights or selecting collection members.
        Default to CUDA mode; choose the platform-specific delegate only when inference is requested.
        """
        if device not in {"cpu", "cuda"}:
            raise ValueError("Feature extraction device must be cpu or cuda")
        self.device = device
        self.landmarker_device = "cpu" if device == "cpu" or platform.system() == "Windows" else "gpu"
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
        Return only facial data; batch results additionally carry provenance and failure reasons.
        """
        return self._extract_result(prepared_image).features or FaceFeatures()

    def _extract_result(self, prepared_image: dict) -> FaceExtractionResult:
        """Associate extracted measurements with their input without deciding pose or image suitability.
        Retain partial BiSeNet evidence and extraction failures for the grouping stage to assess.
        """
        if self._closed:
            raise RuntimeError("Face feature extractor is closed")
        from ailias.imageprocessing.appearance_features import extract_bisenet_features

        original_rgb, head, regions = self._validate_prepared_image(prepared_image)
        result = self._create_extraction_result(prepared_image)
        if head.getchannel("A").getbbox() is None:
            return replace(result, error_type="ValueError", reason="no_head_foreground")
        bisenet = extract_bisenet_features(original_rgb, regions)
        result = replace(result, features=FaceFeatures(bisenet=bisenet))
        detection = self.detect_face_landmarks(head)
        result.execution["mediapipe_executed"] = True
        if len(detection["faces"]) != 1:
            reason = "no_face_detected" if not detection["faces"] else "ambiguous_face_detections"
            return replace(result, error_type="ValueError", reason=reason)
        try:
            features = self._extract_face_measurements(original_rgb, head, regions, detection, bisenet)
        except ValueError as error:
            return replace(result, error_type="ValueError", reason=str(error))
        return replace(result, features=features)

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

    def _create_extraction_result(self, prepared_image: dict) -> FaceExtractionResult:
        """Keep source/frame/backend metadata alongside facial measurements rather than inside them.
        Retain supplied pixels by reference for grouping and review; never reload or alter the source.
        """
        return FaceExtractionResult(
            image=prepared_image.get("original_image"), head_image=prepared_image.get("image"),
            source=prepared_image.get("source"), frame_index=prepared_image.get("frame_index"),
            timestamp_seconds=prepared_image.get("timestamp_seconds"),
            execution={
                "requested_device": self.device, "mediapipe_device": self.landmarker_device,
                "mediapipe_executed": False,
            },
        )

    def _extract_face_measurements(
        self, original_rgb: Image.Image, head: Image.Image, regions: Image.Image, detection: dict,
        bisenet: FaceFeaturesBiseNet,
    ) -> FaceFeatures:
        """Combine MediaPipe points and existing BiSeNet labels into original-pixel feature evidence.
        Preserve raw transform and expression predictions without interpreting orientation or suitability.
        """
        import numpy as np
        from ailias.imageprocessing.appearance_features import photo_features

        landmarks = self._map_face_landmarks(detection["faces"][0], head)
        blendshapes = detection["blendshapes"][0] if detection["blendshapes"] else {}
        transform = detection["transforms"][0] if detection["transforms"] else None
        return FaceFeatures(
            bisenet=bisenet, landmarks={
                "normalized_predicted": landmarks["normalized_predicted"].tolist(),
                "source_xy": landmarks["source_xy"].tolist(), "on_head": landmarks["on_head"].tolist(),
            },
            blendshapes=blendshapes, facial_transform=transform,
            features=photo_features(np.asarray(original_rgb), landmarks["source_xy"], regions,
                                    landmarks["on_head"], region_features=bisenet),
        )

    def _map_face_landmarks(self, normalized_points: list, head: Image.Image) -> dict:
        """Map MediaPipe's 478 normalized predictions onto the unchanged source-image grid.
        Record visible-alpha support without applying quality thresholds, pose limits or selection policy.
        """
        import numpy as np

        normalized = np.asarray(normalized_points, dtype=float)
        if normalized.shape != (MEDIAPIPE_LANDMARK_COUNT, 3) or not np.isfinite(normalized).all():
            raise ValueError("invalid_dense_landmarks")
        source_xy = normalized[:, :2] * head.size
        inside_image = ((source_xy >= 0) & (source_xy < head.size)).all(1)
        sample_pixels = np.clip(np.floor(source_xy).astype(int), 0, np.asarray(head.size) - 1)
        alpha = np.asarray(head.getchannel("A"))
        on_head = inside_image & (alpha[sample_pixels[:, 1], sample_pixels[:, 0]] > 0)
        return {
            "normalized_predicted": normalized, "source_xy": source_xy,
            "on_head": on_head,
        }

    def extract_features_batch(self, prepared_images: Iterable[dict]) -> Iterator[FaceExtractionResult]:
        """Stream facial data together with source metadata and per-image extraction failures.
        Reuse one landmarker and keep original pixels available for subsequent grouping and review copies.
        """
        for prepared in prepared_images:
            if prepared.get("status") == "error":
                yield self._create_error_result(prepared)
                continue
            try:
                yield self._extract_result(prepared)
            except (OSError, ValueError, RuntimeError) as error:
                yield self._create_error_result(prepared, error)

    def _create_error_result(self, prepared_image: dict, error: Exception | None = None) -> FaceExtractionResult:
        """Represent a preparation or extraction failure outside the facial-data class.
        Preserve source identity without claiming successful model execution or fabricated observations.
        """
        return replace(
            self._create_extraction_result(prepared_image), execution={},
            error_type=type(error).__name__ if error is not None else prepared_image.get("error_type"),
            reason=str(error) if error is not None else prepared_image.get("reason"),
        )

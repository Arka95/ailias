"""Image and video-frame preparation with reusable, explicitly supplied models.

The simple path returns head cutouts and facial evidence without writing artifacts.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import math
from pathlib import Path
import platform

from PIL import Image

from ailias.imageprocessing.cleansing import (
    CleansingPolicy, encode_rgb, linear_rgb, white_balance,
)
from ailias.imageprocessing.faceparsing import HEAD_CLASSES
from ailias.imageprocessing.image_io import read_oriented_image
from ailias.imageprocessing.settings import ProcessingPolicy


MODEL_INPUT_SIDE = 512
NEUTRAL_BACKGROUND = (127, 127, 127, 255)
MINIMUM_SUPPORTED_LANDMARKS = 150
MAXIMUM_NEUTRAL_EXPRESSION = 0.5
MAXIMUM_ABSOLUTE_PITCH_ROLL = 25.0
MAXIMUM_ABSOLUTE_YAW = 70.0


@dataclass(frozen=True)
class Frame:
    """Keep decoded RGB pixels with their source and presentation-time position.
    Frame indices refer to decoded frames, not the sampled sequence.
    """

    image: Image.Image
    source: Path
    index: int
    timestamp_seconds: float


class ImageProcessor:
    """Prepare individual images or streamed collections with lazily reused models.
    Call close(), or use a context manager, to release owned inference resources.
    """

    def __init__(
        self, *, modnet_model: str | Path | None = None,
        bisenet_model: str | Path | None = None,
        mediapipe_model: str | Path | None = None,
        device: str = "cuda", gpu_index: int = 0,
        colour_policy: CleansingPolicy | None = None,
        quality_policy: ProcessingPolicy | None = None,
    ):
        """Accept existing model files without loading weights or downloading anything.
        Default to CUDA; explicit CPU mode and supplied colour/quality policies override defaults.
        """
        if device not in {"cpu", "cuda"}:
            raise ValueError("Image processing device must be cpu or cuda")
        if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
            raise ValueError("GPU index must be a nonnegative integer")
        self.device, self.gpu_index = device, gpu_index
        self.landmarker_device = "cpu" if device == "cpu" or platform.system() == "Windows" else "gpu"
        self._model_paths = {
            "modnet": Path(modnet_model) if modnet_model is not None else None,
            "bisenet": Path(bisenet_model) if bisenet_model is not None else None,
            "mediapipe": Path(mediapipe_model) if mediapipe_model is not None else None,
        }
        self._networks = {}
        self._executions = {}
        self._landmarker = None
        self._closed = False
        self.colour_policy = colour_policy if colour_policy is not None else CleansingPolicy()
        self.quality_policy = quality_policy if quality_policy is not None else ProcessingPolicy()

    def __enter__(self):
        """Keep one processor and its model instances alive for the enclosing batch.
        Reject reuse after close rather than silently constructing new runtimes.
        """
        if self._closed:
            raise RuntimeError("Image processor is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release owned model resources when the context ends, including on failure.
        Let the original processing exception propagate to the caller.
        """
        self.close()

    def close(self) -> None:
        """Release cached OpenCV networks and the owned MediaPipe landmarker.
        Closing is idempotent and never shuts down an external model process.
        """
        self._closed = True
        self._networks.clear()
        self._executions.clear()
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            landmarker.close()

    @staticmethod
    def load_image(path: str | Path) -> Image.Image:
        """Decode detached RGB/RGBA pixels with EXIF orientation and ICC colour handling.
        Leave the source untouched; profile conversion is not white-balance correction.
        """
        return read_oriented_image(Path(path), preserve_alpha=True)[0]

    @staticmethod
    def crop_image(image: Image.Image, bounds: tuple[int, int, int, int]) -> Image.Image:
        """Crop explicit left/top/right/bottom pixel bounds without resizing the result.
        Reject out-of-image or empty bounds rather than introducing padded pixels.
        """
        if (len(bounds) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in bounds)):
            raise ValueError("Crop bounds must contain four integer pixel coordinates")
        left, top, right, bottom = bounds
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError("Crop bounds must describe a nonempty region inside the image")
        return image.crop(bounds)

    @staticmethod
    def extract_video_frames(
        path: str | Path, *, interval_seconds: float = 1.0, max_frames: int | None = None,
    ) -> Iterator[Frame]:
        """Stream sampled RGB frames using presentation timestamps, not average FPS.
        Reject missing/nonmonotonic timing and rotated sources; close the iterator early to release decoding.
        """
        if (isinstance(interval_seconds, bool) or not math.isfinite(interval_seconds)
                or interval_seconds <= 0):
            raise ValueError("Frame interval must be finite and positive")
        if max_frames is not None and (
            isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
        ):
            raise ValueError("Frame limit must be a positive integer")
        import av

        source = Path(path)
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise ValueError("Input has no video stream")
            stream = container.streams.video[0]
            first, previous, next_time, count = None, None, 0.0, 0
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Video frame lacks a presentation timestamp")
                absolute = float(frame.pts * frame.time_base)
                if not math.isfinite(absolute) or (previous is not None and absolute <= previous):
                    raise ValueError("Video presentation timestamps must be finite and increasing")
                previous = absolute
                if first is None:
                    first = absolute
                relative = absolute - first
                if relative + 1e-8 < next_time:
                    continue
                rotation = float(getattr(frame, "rotation", 0) or stream.metadata.get("rotate", 0))
                if not math.isfinite(rotation) or rotation % 360:
                    raise ValueError("Rotated video requires an explicitly normalized source copy")
                yield Frame(Image.fromarray(frame.to_ndarray(format="rgb24")), source, index, relative)
                count += 1
                if max_frames is not None and count >= max_frames:
                    return
                next_time = (math.floor(relative / interval_seconds + 1e-8) + 1) * interval_seconds
            if not count:
                raise ValueError("Video decoded no frames")

    def _load_network(self, name: str):
        """Load and cache the MODNet or BiSeNet ONNX CNN on the explicitly selected backend.
        Require local weights and available CUDA when requested; never install, download or retry on CPU.
        """
        if self._closed:
            raise RuntimeError("Image processor is closed")
        if name not in self._networks:
            path = self._model_paths[name]
            if path is None:
                raise ValueError(f"Supply the existing {name} model path")
            if not path.is_file():
                raise FileNotFoundError(path)
            import cv2

            try:
                if self.device == "cuda":
                    from ailias.imageprocessing.cuda import require_cuda

                    require_cuda(cv2)
                    cv2.cuda.setDevice(self.gpu_index)
                network = cv2.dnn.readNetFromONNX(str(path))
                network.setPreferableBackend(
                    cv2.dnn.DNN_BACKEND_CUDA if self.device == "cuda" else cv2.dnn.DNN_BACKEND_OPENCV)
                network.setPreferableTarget(
                    cv2.dnn.DNN_TARGET_CUDA if self.device == "cuda" else cv2.dnn.DNN_TARGET_CPU)
            except cv2.error as error:
                raise ValueError(f"Unable to load the {name} ONNX model") from error
            self._networks[name] = network
        return self._networks[name]

    def _run_network(self, name: str, values):
        """Execute the cached MODNet/BiSeNet CNN and verify CUDA graph use after its first forward.
        Refuse OpenCV CPU fallback in CUDA mode and keep backend execution evidence in memory.
        """
        import cv2

        network = self._load_network(name)
        try:
            network.setInput(values)
            prediction = network.forward()
        except cv2.error as error:
            raise ValueError(f"{'MODNet' if name == 'modnet' else 'BiSeNet'} inference failed") from error
        if name not in self._executions:
            if self.device == "cuda":
                from ailias.imageprocessing.cuda import graph_execution

                self._executions[name] = {
                    **graph_execution(network.dump()), "device": f"cuda:{self.gpu_index}"}
            else:
                self._executions[name] = {"status": "completed", "device": "cpu"}
        return prediction

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Use the MODNet portrait-matting CNN to predict soft person/background opacity.
        Letterbox to 512 square, then restore alpha to unchanged source pixels and preserve existing transparency.
        """
        import cv2
        import numpy as np

        result = image.convert("RGBA")
        composite = Image.new("RGBA", image.size, NEUTRAL_BACKGROUND)
        composite.alpha_composite(result)
        scale = MODEL_INPUT_SIDE / max(image.size)
        width, height = (max(1, round(value * scale)) for value in image.size)
        x, y = (MODEL_INPUT_SIDE - width) // 2, (MODEL_INPUT_SIDE - height) // 2
        values = np.full((MODEL_INPUT_SIDE, MODEL_INPUT_SIDE, 3), 127.5, dtype=np.float32)
        values[y:y + height, x:x + width] = cv2.resize(
            np.asarray(composite.convert("RGB")), (width, height), interpolation=cv2.INTER_AREA)
        prediction = self._run_network(
            "modnet", cv2.dnn.blobFromImage(values, 1 / 127.5, mean=(127.5,) * 3))
        if prediction.shape != (1, 1, MODEL_INPUT_SIDE, MODEL_INPUT_SIDE) or not np.isfinite(prediction).all():
            raise ValueError("MODNet returned invalid opacity")
        alpha = cv2.resize(prediction[0, 0, y:y + height, x:x + width],
                           image.size, interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0, 1) * np.asarray(result.getchannel("A"))
        result.putalpha(Image.fromarray(np.rint(alpha).astype(np.uint8)))
        return result

    def isolate_head(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Use the BiSeNet ResNet18 CNN to label face parts and retain face, ears and hair.
        Infer on a grey composite, then restrict existing alpha without cropping or restoring removed pixels.
        """
        import cv2
        import numpy as np

        rgba = image.convert("RGBA")
        composite = Image.new("RGBA", image.size, NEUTRAL_BACKGROUND)
        composite.alpha_composite(rgba)
        values = cv2.resize(np.asarray(composite.convert("RGB")), (MODEL_INPUT_SIDE,) * 2,
                            interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        values = ((values - np.array([0.485, 0.456, 0.406], np.float32))
                  / np.array([0.229, 0.224, 0.225], np.float32))
        prediction = self._run_network("bisenet", values.transpose(2, 0, 1)[None])
        if (prediction.shape != (1, 19, MODEL_INPUT_SIDE, MODEL_INPUT_SIDE)
                or not np.isfinite(prediction).all()):
            raise ValueError("BiSeNet returned invalid region scores")
        labels = cv2.resize(prediction[0].argmax(0).astype(np.uint8), image.size,
                            interpolation=cv2.INTER_NEAREST)
        alpha = np.asarray(rgba.getchannel("A")).copy()
        labels[alpha == 0] = 0
        alpha[~np.isin(labels, HEAD_CLASSES)] = 0
        rgba.putalpha(Image.fromarray(alpha))
        return rgba, Image.fromarray(labels)

    def detect_face_landmarks(self, image: Image.Image) -> dict:
        """Use MediaPipe's face-detection/landmark bundle on neutral-grey RGB for points and expressions.
        Windows CUDA mode uses CPU because MediaPipe's Python GPU delegate is unsupported on Windows.
        Other CUDA platforms request GPU; explicit CPU mode stays on CPU, without retry fallback.
        """
        import numpy as np

        if self._closed:
            raise RuntimeError("Image processor is closed")
        if self._landmarker is None:
            path = self._model_paths["mediapipe"]
            if path is None:
                raise ValueError("Supply the existing mediapipe model path")
            if not path.is_file():
                raise FileNotFoundError(path)
            from ailias.imageprocessing.dense import DenseDetector

            self._landmarker = DenseDetector(path, device=self.landmarker_device)
        composite = Image.new("RGBA", image.size, NEUTRAL_BACKGROUND)
        composite.alpha_composite(image.convert("RGBA"))
        return self._landmarker.detect(np.asarray(composite.convert("RGB")))

    def normalize_colour(
        self, image: Image.Image, *, excluded_bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[Image.Image, dict]:
        """Estimate bounded white-balance gains from scene neutrals, excluding the supplied head bounds.
        Use classical linear-RGB correction, not a CNN; preserve alpha and skip unsupported colour casts.
        """
        import numpy as np

        boxes = []
        if excluded_bounds is not None:
            left, top, right, bottom = excluded_bounds
            boxes.append([left, top, right - left, bottom - top])
        rgb = image.convert("RGB")
        balance = white_balance(rgb, boxes, self.colour_policy)
        corrected = encode_rgb(linear_rgb(np.asarray(rgb)) * np.asarray(balance["gains"]))
        if "A" in image.getbands():
            corrected = corrected.convert("RGBA")
            corrected.putalpha(image.getchannel("A"))
        return corrected, balance

    def process_image(
        self, source: str | Path | Image.Image | Frame, *, normalize_colour: bool = True,
    ) -> dict:
        """Run MODNet, BiSeNet and MediaPipe in order, then extract original-pixel facial features.
        Return native-grid images, predictions and quality flags in memory; never crop, align or write sidecars.
        """
        import numpy as np
        from ailias.imageprocessing.appearance_features import photo_features
        from ailias.imageprocessing.features import face_quality

        frame = source if isinstance(source, Frame) else None
        image = frame.image if frame is not None else source
        image = image.copy() if isinstance(image, Image.Image) else self.load_image(image)
        original_rgb = image.convert("RGB")
        head, regions = self.isolate_head(self.remove_background(image))
        bounds = head.getchannel("A").getbbox()
        result = {
            "status": "review", "image": head, "regions": regions, "features": None,
            "landmarks": None, "blendshapes": {}, "facial_transform": None,
            "quality": {"reasons": []}, "colour": {"status": "not_applied"},
            "source": frame.source if frame is not None else (
                None if isinstance(source, Image.Image) else Path(source)),
            "frame_index": frame.index if frame is not None else None,
            "timestamp_seconds": frame.timestamp_seconds if frame is not None else None,
            "features_colour_source": "original_rgb",
            "metric_depth_measured": False,
            "execution": {
                "requested_device": self.device, "networks": dict(self._executions),
                "mediapipe_device": self.landmarker_device, "mediapipe_executed": False,
            },
        }
        if bounds is None:
            result["quality"]["reasons"].append("no_head_foreground")
            return result
        detection = self.detect_face_landmarks(head)
        result["execution"]["mediapipe_executed"] = True
        if normalize_colour:
            corrected, result["colour"] = self.normalize_colour(original_rgb, excluded_bounds=bounds)
            corrected = corrected.convert("RGBA")
            corrected.putalpha(head.getchannel("A"))
            result["image"] = corrected
        if len(detection["faces"]) != 1:
            result["quality"]["reasons"].append(
                "no_face_detected" if not detection["faces"] else "multiple_faces_detected")
            return result
        landmarks = np.asarray(detection["faces"][0], dtype=float)
        if landmarks.shape != (478, 3) or not np.isfinite(landmarks).all():
            result["quality"]["reasons"].append("invalid_dense_landmarks")
            return result
        xy = landmarks[:, :2] * image.size
        inside = ((xy >= 0) & (xy < image.size)).all(1)
        pixels = np.clip(np.floor(xy).astype(int), 0, np.asarray(image.size) - 1)
        on_head = inside & (np.asarray(head.getchannel("A"))[pixels[:, 1], pixels[:, 0]] >= 240)
        low, high = xy[:468].min(0), xy[:468].max(0)
        left, top = np.maximum(0, low)
        right, bottom = np.minimum(image.size, high)
        if right <= left or bottom <= top:
            result["quality"]["reasons"].append("face_outside_image")
            return result
        quality = face_quality(original_rgb, [left, top, right - left, bottom - top], self.quality_policy)
        if not inside[:468].all():
            quality["reasons"].append("face_clipped")
        if int(on_head[:468].sum()) < MINIMUM_SUPPORTED_LANDMARKS:
            quality["reasons"].append("insufficient_head_support")
        blendshapes = detection["blendshapes"][0] if detection["blendshapes"] else {}
        expression = max((value for name, value in blendshapes.items()
                          if name.startswith(("mouth", "brow", "cheek", "noseSneer", "eyeWide", "eyeSquint"))
                          or name == "jawOpen"), default=0.0)
        if expression > MAXIMUM_NEUTRAL_EXPRESSION:
            quality["reasons"].append("non_neutral_expression")
        transform = detection["transforms"][0] if detection["transforms"] else None
        matrix = np.asarray(transform, dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=.02)
                or not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1.0, abs_tol=.02)):
            quality["reasons"].append("missing_or_invalid_pose")
        else:
            yaw = math.degrees(math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0])))
            pitch = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
            roll = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
            quality["pose"] = {"pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll,
                               "calibrated": False}
            if (abs(yaw) > MAXIMUM_ABSOLUTE_YAW
                    or max(abs(pitch), abs(roll)) > MAXIMUM_ABSOLUTE_PITCH_ROLL):
                quality["reasons"].append("extreme_pose")
        result.update(
            status="review" if quality["reasons"] else "ready", quality=quality,
            landmarks={"normalized_predicted": landmarks.tolist(), "source_xy": xy.tolist(),
                       "on_head": on_head.tolist()},
            blendshapes=blendshapes,
            facial_transform=transform,
            features=photo_features(np.asarray(original_rgb), xy, regions, on_head),
        )
        return result

    def process_images(
        self, sources: Iterable[str | Path | Image.Image | Frame], *, normalize_colour: bool = True,
    ) -> Iterator[dict]:
        """Yield one processed result at a time while reusing the owned model instances.
        Return per-image failures in memory and continue, without retaining a collection or writing files.
        """
        for source in sources:
            try:
                yield self.process_image(source, normalize_colour=normalize_colour)
            except (OSError, ValueError, RuntimeError) as error:
                yield {"status": "error", "error_type": type(error).__name__, "reason": str(error),
                       "source": source.source if isinstance(source, Frame) else (
                           source if isinstance(source, (str, Path)) else None)}

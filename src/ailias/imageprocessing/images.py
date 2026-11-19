"""Image and video-frame preparation with reusable, explicitly supplied models.

The simple path returns prepared head images and labels without writing artifacts.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import math
from pathlib import Path

from PIL import Image

from ailias.imageprocessing.cleansing import (
    CleansingPolicy, encode_rgb, linear_rgb, white_balance,
)
from ailias.imageprocessing.faceparsing import HEAD_CLASSES
from ailias.imageprocessing.image_io import read_oriented_image


MODNET_INPUT_SIDE = 512
BISENET_INPUT_SIDE = 512
BISENET_CLASS_COUNT = 19
# Trained model preprocessing, not adjustable image-quality thresholds.
MODNET_RGB_CENTER = 127.5
BISENET_RGB_MEAN = (0.485, 0.456, 0.406)
BISENET_RGB_STANDARD_DEVIATION = (0.229, 0.224, 0.225)
RGB_MAX_VALUE = 255
NEUTRAL_BACKGROUND = (127, 127, 127, 255)


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
        device: str = "cuda", gpu_index: int = 0,
        colour_policy: CleansingPolicy | None = None,
    ):
        """Accept existing model files without loading weights or downloading anything.
        Default to CUDA and the current colour policy unless explicitly supplied.
        """
        if device not in {"cpu", "cuda"}:
            raise ValueError("Image processing device must be cpu or cuda")
        if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
            raise ValueError("GPU index must be a nonnegative integer")
        self.device, self.gpu_index = device, gpu_index
        self._model_paths = {
            "modnet": Path(modnet_model) if modnet_model is not None else None,
            "bisenet": Path(bisenet_model) if bisenet_model is not None else None,
        }
        self._networks = {}
        self._executions = {}
        self._closed = False
        self.colour_policy = colour_policy if colour_policy is not None else CleansingPolicy()

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
        """Release the owned MODNet and BiSeNet OpenCV networks.
        Closing is idempotent and never shuts down an external model process.
        """
        self._closed = True
        self._networks.clear()
        self._executions.clear()

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
        scale = MODNET_INPUT_SIDE / max(image.size)
        width, height = (max(1, round(value * scale)) for value in image.size)
        x, y = (MODNET_INPUT_SIDE - width) // 2, (MODNET_INPUT_SIDE - height) // 2
        values = np.full((MODNET_INPUT_SIDE, MODNET_INPUT_SIDE, 3), MODNET_RGB_CENTER, dtype=np.float32)
        values[y:y + height, x:x + width] = cv2.resize(
            np.asarray(composite.convert("RGB")), (width, height), interpolation=cv2.INTER_AREA)
        prediction = self._run_network(
            "modnet", cv2.dnn.blobFromImage(values, 1 / MODNET_RGB_CENTER, mean=(MODNET_RGB_CENTER,) * 3))
        if prediction.shape != (1, 1, MODNET_INPUT_SIDE, MODNET_INPUT_SIDE) or not np.isfinite(prediction).all():
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
        values = cv2.resize(np.asarray(composite.convert("RGB")), (BISENET_INPUT_SIDE,) * 2,
                            interpolation=cv2.INTER_LINEAR).astype(np.float32) / RGB_MAX_VALUE
        values = ((values - np.array(BISENET_RGB_MEAN, np.float32))
                  / np.array(BISENET_RGB_STANDARD_DEVIATION, np.float32))
        prediction = self._run_network("bisenet", values.transpose(2, 0, 1)[None])
        if (prediction.shape != (1, BISENET_CLASS_COUNT, BISENET_INPUT_SIDE, BISENET_INPUT_SIDE)
                or not np.isfinite(prediction).all()):
            raise ValueError("BiSeNet returned invalid region scores")
        labels = cv2.resize(prediction[0].argmax(0).astype(np.uint8), image.size,
                            interpolation=cv2.INTER_NEAREST)
        alpha = np.asarray(rgba.getchannel("A")).copy()
        labels[alpha == 0] = 0
        alpha[~np.isin(labels, HEAD_CLASSES)] = 0
        rgba.putalpha(Image.fromarray(alpha))
        return rgba, Image.fromarray(labels)

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
        """Prepare a native-grid head with MODNet, BiSeNet and optional colour normalization.
        Return original pixels and existing labels for separate feature extraction; never crop or write sidecars.
        """
        image = self._load_source_image(source)
        original_rgb = image.convert("RGB")
        foreground = self.remove_background(image)
        head, regions = self.isolate_head(foreground)
        result = self._create_processing_result(source, original_rgb, head, regions)

        head_bounds = head.getchannel("A").getbbox()
        if head_bounds is None:
            result["quality"]["reasons"].append("no_head_foreground")
            return result

        if normalize_colour:
            corrected, result["colour"] = self.normalize_colour(
                original_rgb, excluded_bounds=head_bounds)
            corrected = corrected.convert("RGBA")
            corrected.putalpha(head.getchannel("A"))
            result["image"] = corrected

        result["status"] = "ready"
        return result

    def _load_source_image(self, source: str | Path | Image.Image | Frame) -> Image.Image:
        """Resolve a file, decoded image or video frame into independently owned pixels.
        Copy supplied images so preparation cannot modify the caller's original.
        """
        if isinstance(source, Frame):
            return source.image.copy()
        if isinstance(source, Image.Image):
            return source.copy()
        return self.load_image(source)

    def _create_processing_result(
        self, source: str | Path | Image.Image | Frame, original_rgb: Image.Image,
        head: Image.Image, regions: Image.Image,
    ) -> dict:
        """Assemble preparation products with untouched RGB for later colour measurements.
        Preserve source/frame identity and backend evidence without owning facial analysis.
        """
        source_path, frame_index, timestamp = None, None, None
        if isinstance(source, Frame):
            source_path, frame_index, timestamp = source.source, source.index, source.timestamp_seconds
        elif isinstance(source, (str, Path)):
            source_path = Path(source)
        return {
            "status": "review", "image": head, "original_image": original_rgb, "regions": regions,
            "quality": {"reasons": []}, "colour": {"status": "not_applied"},
            "source": source_path, "frame_index": frame_index, "timestamp_seconds": timestamp,
            "execution": {
                "requested_device": self.device, "networks": dict(self._executions),
            },
        }

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

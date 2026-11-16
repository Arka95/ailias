"""Define the image-analysis backend contract and the CPU OpenCV implementation.

Provides face detection/recognition and person segmentation, not head reconstruction.
"""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Protocol

from PIL import Image

from ailias.imageprocessing.models import REVISION, verified_models
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.storage import Store


@dataclass
class Face:
    box: list[float]
    landmarks: list[list[float]]
    score: float
    embedding: list[float]


class Backend(Protocol):
    provenance: dict

    def detect(self, image: Image.Image) -> list[Face]: ...

    def segment(self, image: Image.Image) -> Image.Image: ...


def detect_scaled_rows(cv, original, maximum, infer):
    height, width = original.shape[:2]
    attempted = set()
    for side in (maximum, min(maximum, 960), min(maximum, 640), min(maximum, 320)):
        scale = min(1., side / max(width, height))
        size = max(1, round(width*scale)), max(1, round(height*scale))
        if size in attempted:
            continue
        attempted.add(size)
        small = cv.resize(original, size)
        rows = infer(small)
        if rows is None or not len(rows):
            continue
        mapped = []
        for row in rows:
            row = row.copy()
            row[:14:2] *= width/size[0]
            row[1:14:2] *= height/size[1]
            mapped.append(row)
        return mapped
    return []


class OpenCVBackend:
    def __init__(self, store: Store, policy: ProcessingPolicy, body: bool = True):
        paths = verified_models(store, body)
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise ValueError('Install optional dependencies: pip install -e ".[imageprocessing]"') from error
        self.cv, self.np, self.policy = cv2, np, policy
        cv2.setNumThreads(policy.cpu_threads)
        self.detector = cv2.FaceDetectorYN.create(
            str(paths["yunet"]), "", (320, 320), policy.detection_score, 0.3, 5000,
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(paths["sface"]), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )
        self.segmenter = cv2.dnn.readNetFromONNX(str(paths["pphumanseg"])) if body else None
        if self.segmenter is not None:
            self.segmenter.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.segmenter.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.provenance = {
            "provider": "opencv/opencv_zoo", "revision": REVISION,
            "device": "cpu", "dtype": "float32",
            "opencv": cv2.__version__, "numpy": np.__version__,
            "pillow": version("Pillow"), "cpu_threads": policy.cpu_threads,
            "models": {name: str(path.relative_to(store.root)) for name, path in paths.items()},
            "detection_scales": "configured side; no-face-only 960/640/320 retries, unchanged threshold",
        }

    def detect(self, image):
        cv, np = self.cv, self.np
        original = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
        def infer(small):
            self.detector.setInputSize((small.shape[1], small.shape[0]))
            return self.detector.detect(small)[1]
        try:
            rows = detect_scaled_rows(cv, original, self.policy.detection_max_side, infer)
            faces = []
            for mapped in rows:
                aligned = self.recognizer.alignCrop(original, mapped)
                feature = self.recognizer.feature(aligned).flatten()
                feature /= max(float(np.linalg.norm(feature)), 1e-12)
                faces.append(Face(mapped[:4].tolist(), mapped[4:14].reshape(5, 2).tolist(),
                                  float(mapped[14]), feature.tolist()))
            return sorted(faces, key=lambda face: (face.box[0], face.box[1]))
        except cv.error as error:
            raise ValueError("OpenCV face inference failed") from error

    def segment(self, image):
        if self.segmenter is None:
            raise ValueError("Body segmentation is disabled")
        cv, np = self.cv, self.np
        reduced = image.copy()
        reduced.thumbnail((self.policy.detection_max_side, self.policy.detection_max_side))
        rgb = np.asarray(reduced)
        blob = cv.dnn.blobFromImage(rgb, 1 / 127.5, (192, 192), (127.5,) * 3)
        try:
            self.segmenter.setInput(blob)
            scores = self.segmenter.forward()[0].transpose(1, 2, 0)
            scores = cv.resize(scores, reduced.size, interpolation=cv.INTER_LINEAR)
            labels = np.argmax(scores, axis=2).astype(np.uint8) * 255
            return Image.fromarray(labels).resize(image.size, Image.Resampling.NEAREST)
        except cv.error as error:
            raise ValueError("OpenCV person segmentation failed") from error

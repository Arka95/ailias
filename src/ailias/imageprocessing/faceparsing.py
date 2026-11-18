"""Semantic head extraction: keep facial parts/hair, not the whole person."""

import json
from pathlib import Path

from PIL import Image

from ailias.imageprocessing.models import download
from ailias.storage import atomic_json, digest

REVISION = "4f0632b17a3e729a7871683aab7c89b6a595da9f"
SOURCE = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx"
SHA256 = "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"
DIRECTORY = str(Path("imageprocessing") / "models" / "face-parsing-resnet18-v1")
# CelebAMask-HQ: skin, brows, eyes, glasses, ears, nose, mouth/lips, hair.
HEAD_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17)
EXCLUDED_CLASSES = (9, 14, 15, 16, 18)


def head_context(bounds, size):
    x, y, width, height = bounds
    half_width = max(width, height*0.9)
    return (max(0, int(x + width/2 - half_width)),
            max(0, int(y - height*0.7)),
            min(size[0], round(x + width/2 + half_width)),
            min(size[1], round(y + height*1.1)))


def install_model(store):
    root = store.resolve(DIRECTORY)
    target = root / "resnet18.onnx"
    if not target.is_file() or digest(target) != SHA256:
        download(SOURCE, target, 53205364, SHA256)
    license_path = root / "LICENSE.txt"
    license_source = f"https://raw.githubusercontent.com/yakhyo/face-parsing/{REVISION}/LICENSE"
    if not license_path.is_file():
        download(license_source, license_path, 65536)
    record = {
        "provider": "yakhyo/face-parsing", "revision": REVISION, "license": "MIT",
        "source": SOURCE, "model": store.reference_artifact(target),
        "license_artifact": store.reference_artifact(license_path),
        "license_source": license_source, "classes": 19, "head_classes": list(HEAD_CLASSES),
    }
    atomic_json(root / "provenance.json", record)
    return record


class HeadParser:
    def __init__(self, store, device, gpu_index=0):
        import cv2
        import numpy as np

        self.cv, self.np, self.network = cv2, np, None
        root = store.resolve(DIRECTORY)
        path = root / "resnet18.onnx"
        if not path.is_file() or digest(path) != SHA256:
            raise ValueError("Missing/unverified head-parsing model; run imageprocessing models")
        record = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        if not store.verify(record["license_artifact"]):
            raise ValueError("Head-parsing license/provenance changed; reinstall models")
        self.network = cv2.dnn.readNetFromONNX(str(path))
        self.device = device
        if device == "cuda":
            from ailias.imageprocessing.cuda import require_cuda
            require_cuda(cv2)
            cv2.cuda.setDevice(gpu_index)
            self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.provenance = {
            "provider": record["provider"], "revision": REVISION, "model": store.reference_artifact(path),
            "license_artifact": record["license_artifact"],
            "device": f"cuda:{gpu_index}" if device == "cuda" else "cpu", "dtype": "float32",
            "head_classes": list(HEAD_CLASSES), "execution": {"status": "not_executed"},
            "preprocessing": "detected-face context; RGB bilinear 512; ImageNet mean/std",
            "output_geometry": "argmax labels restored with nearest neighbour to unchanged native canvas",
        }

    def parse(self, image, bounds):
        cv, np = self.cv, self.np
        region = head_context(bounds, image.size)
        portrait = image.crop(region).convert("RGB")
        values = cv.resize(np.asarray(portrait), (512, 512),
                           interpolation=cv.INTER_LINEAR).astype(np.float32)/255
        values = ((values - np.array([0.485, 0.456, 0.406], np.float32))
                  / np.array([0.229, 0.224, 0.225], np.float32))
        try:
            self.network.setInput(values.transpose(2, 0, 1)[None])
            output = self.network.forward()
            if output.shape != (1, 19, 512, 512) or not np.isfinite(output).all():
                raise ValueError("Face parser returned invalid class scores")
            if self.provenance["execution"]["status"] == "not_executed":
                if self.device == "cuda":
                    from ailias.imageprocessing.cuda import graph_execution
                    self.provenance["execution"] = graph_execution(self.network.dump())
                else:
                    self.provenance["execution"] = {"status": "completed", "backend": "OpenCV CPU"}
            labels = output[0].argmax(0).astype(np.uint8)
            restored = Image.new("L", image.size)
            restored.paste(Image.fromarray(cv.resize(
                labels, portrait.size, interpolation=cv.INTER_NEAREST)), region[:2])
            return restored
        except cv.error as error:
            if self.device == "cuda":
                from ailias.imageprocessing.cuda import CudaUnavailable
                raise CudaUnavailable("Head parsing CUDA inference failed; no CPU retry") from error
            raise ValueError("Head-parsing inference failed") from error

    def close(self):
        self.network = None


def head_alpha(alpha, parts, bounds):
    import cv2
    import numpy as np

    from ailias.imageprocessing.cleansing import core_slice

    if not isinstance(alpha, Image.Image) or alpha.mode != "L":
        raise ValueError("Person matting must provide a grayscale alpha image")
    if not isinstance(parts, Image.Image) or parts.mode != "L" or parts.size != alpha.size:
        raise ValueError("Head labels must match native portrait coordinates")
    labels = np.asarray(parts)
    if labels.max() > 18:
        raise ValueError("Unknown face-parsing label")
    selected = np.isin(labels, HEAD_CLASSES).astype(np.uint8)
    _, components = cv2.connectedComponents(selected, connectivity=8)
    scores = np.bincount(components[core_slice(bounds, parts.size)].ravel(),
                         minlength=int(components.max()) + 1)
    scores[0] = 0
    subject = components == scores.argmax() if scores.max() else np.zeros_like(selected, dtype=bool)
    left, top, right, bottom = head_context(bounds, parts.size)
    # Grow only hair edges, never facial skin into the adjacent neck.
    radius = max(1, round(max(right - left, bottom - top)/512*2))
    fringe = cv2.dilate((subject & (labels == 17)).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius + 1, 2*radius + 1)))
    allowed = subject | fringe.astype(bool)
    allowed[np.isin(labels, EXCLUDED_CLASSES)] = False
    clipped = bottom < parts.height and subject[bottom - 1, left:right].mean() > 0.05
    allowed[:top] = False
    allowed[bottom:] = False
    allowed[:, :left] = False
    allowed[:, right:] = False
    values = np.asarray(alpha).copy()
    values[~allowed] = 0
    return Image.fromarray(values), {
        "target": "head", "kept": "face, eyes/brows, nose, mouth, ears and hair",
        "excluded": "neck, torso/clothes, hats, jewellery and background",
        "semantic_foreground_fraction": float(subject.mean()),
        "identity_component": "largest face-core overlap; not all heads in the image",
        "parser_context": [left, top, right, bottom],
        "review_reason": "semantic_head_reaches_lower_context_boundary" if clipped else None,
    }

"""Pinned portrait matting; conversion tools are used only during explicit setup."""

import json
import os
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from importlib import import_module
from pathlib import Path

from PIL import Image

from ailias.imageprocessing.models import download
from ailias.storage import atomic_json, digest

REVISION = "28165a451e4610c9d77cfdf925a94610bb2810fb"
SOURCE = ("https://drive.usercontent.google.com/download?"
          "id=1cgycTQlYXpTh26gB9FTnthE7AvruV8hd&export=download&confirm=t")
SOURCE_SHA256 = "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9"
DIRECTORY = str(Path("imageprocessing") / "models" / "modnet-photographic-v1")
RECIPE = "modnet-onnxsim-static-512-v1"
INPUT_SIDE = 512


def verified_model(store):
    root = store.resolve(DIRECTORY)
    record = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if (record.get("source_sha256") != SOURCE_SHA256 or record.get("recipe") != RECIPE
            or not record.get("equivalence_checked") or not store.verify(record["model"])
            or not store.verify(record["license_artifact"])):
        raise ValueError("Matting model changed; run imageprocessing models again")
    return store.artifact_path(record["model"]), record


def install_model(store):
    root = store.resolve(DIRECTORY)
    if (root / "provenance.json").is_file():
        try:
            return verified_model(store)[1]
        except (OSError, ValueError, KeyError):
            pass
    try:
        import onnx
        import_module("onnxruntime")  # Prevent onnxsim from installing a runtime.
        import onnxsim
    except ImportError as error:
        raise ValueError("Model setup needs the optional matting-tools extra; "
                         "install with python -m pip install -e .[matting-tools]") from error
    root.mkdir(parents=True, exist_ok=True)
    original = root / "modnet.onnx"
    temporary = root / "modnet-512.preparing.onnx"
    try:
        if not original.is_file() or digest(original) != SOURCE_SHA256:
            download(SOURCE, original, 25888640, SOURCE_SHA256)
        source = onnx.load(str(original))
        with redirect_stdout(sys.stderr):
            converted, valid = onnxsim.simplify(
                source, overwrite_input_shapes={source.graph.input[0].name: [1, 3, 512, 512]},
                check_n=1)
        if not valid:
            raise ValueError("Static matting model failed source equivalence check")
        onnx.save(converted, str(temporary))
        target = root / "modnet-512.onnx"
        os.replace(temporary, target)
        license_path = root / "LICENSE.txt"
        license_source = f"https://raw.githubusercontent.com/ZHKKKe/MODNet/{REVISION}/LICENSE"
        download(license_source, license_path, 65536)
        record = {
            "provider": "ZHKKKe/MODNet", "revision": REVISION, "license": "Apache-2.0",
            "source": SOURCE, "source_sha256": SOURCE_SHA256, "recipe": RECIPE,
            "source_documentation": f"https://github.com/ZHKKKe/MODNet/blob/{REVISION}/onnx/README.md",
            "model": store.reference_artifact(target), "input_shape": [1, 3, 512, 512],
            "license_artifact": store.reference_artifact(license_path),
            "license_source": license_source, "equivalence_checked": True,
            "preparation_tools": {name: version(name) for name in ("onnx", "onnxsim", "onnxruntime")},
        }
        atomic_json(root / "provenance.json", record)
        original.unlink()
        return record
    finally:
        temporary.unlink(missing_ok=True)


class MattingBackend:
    def __init__(self, store, device="cpu", cpu_threads=2, gpu_index=0):
        import cv2
        import numpy as np

        self.cv, self.np, self.network, self.parser = cv2, np, None, None
        if device not in {"cpu", "cuda"}:
            raise ValueError("Matting device must be cpu or cuda")
        try:
            path, record = verified_model(store)
        except FileNotFoundError as error:
            raise ValueError("Missing matting model; run imageprocessing models first") from error
        cv2.setNumThreads(cpu_threads)
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
        from ailias.imageprocessing.faceparsing import HeadParser
        self.parser = HeadParser(store, device, gpu_index)
        self.provenance = {
            "provider": record["provider"], "revision": REVISION, "recipe": RECIPE,
            "model": record["model"], "device": f"cuda:{gpu_index}" if device == "cuda" else "cpu",
            "dtype": "float32", "opencv": cv2.__version__, "input_side": INPUT_SIDE,
            "resize": "aspect-preserving letterbox; alpha restored to original crop pixels",
            "execution": {"status": "not_executed"},
            "head_parsing": self.parser.provenance,
        }

    def matte(self, image):
        cv, np = self.cv, self.np
        scale = INPUT_SIDE / max(image.size)
        width, height = max(1, round(image.width*scale)), max(1, round(image.height*scale))
        x, y = (INPUT_SIDE - width)//2, (INPUT_SIDE - height)//2
        values = np.full((INPUT_SIDE, INPUT_SIDE, 3), 127.5, dtype=np.float32)
        values[y:y + height, x:x + width] = cv.resize(
            np.asarray(image.convert("RGB")), (width, height), interpolation=cv.INTER_AREA)
        try:
            self.network.setInput(cv.dnn.blobFromImage(values, 1/127.5, mean=(127.5,)*3))
            prediction = self.network.forward()
            if prediction.shape != (1, 1, INPUT_SIDE, INPUT_SIDE) or not np.isfinite(prediction).all():
                raise ValueError("Matting model returned invalid alpha")
            if self.provenance["execution"]["status"] == "not_executed":
                if self.device == "cuda":
                    from ailias.imageprocessing.cuda import graph_execution
                    self.provenance["execution"] = graph_execution(self.network.dump())
                else:
                    self.provenance["execution"] = {"status": "completed", "backend": "OpenCV CPU"}
            alpha = cv.resize(prediction[0, 0, y:y + height, x:x + width],
                              image.size, interpolation=cv.INTER_LINEAR)
            return Image.fromarray(np.rint(np.clip(alpha, 0, 1)*255).astype(np.uint8))
        except cv.error as error:
            if self.device == "cuda":
                from ailias.imageprocessing.cuda import CudaUnavailable
                raise CudaUnavailable("Matting CUDA inference failed; no CPU retry") from error
            raise ValueError("Matting inference failed") from error

    def close(self):
        self.network = None
        if self.parser is not None:
            self.parser.close()

    def parse(self, image, bounds):
        return self.parser.parse(image, bounds)

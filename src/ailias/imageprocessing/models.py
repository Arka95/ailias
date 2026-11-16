"""Explicitly download pinned image-analysis models and verify local model integrity."""

import os
import tempfile
import urllib.request
from pathlib import Path

from ailias.storage import Store, atomic_json, digest

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODEL_SPECS = {
    "yunet": {
        "directory": "face_detection_yunet",
        "filename": "face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "size": 232589, "license": "MIT",
    },
    "sface": {
        "directory": "face_recognition_sface",
        "filename": "face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "size": 38696353, "license": "Apache-2.0",
    },
    "pphumanseg": {
        "directory": "human_segmentation_pphumanseg",
        "filename": "human_segmentation_pphumanseg_2023mar.onnx",
        "sha256": "552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24",
        "size": 6163938, "license": "Apache-2.0",
    },
}


def model_paths(store: Store, body: bool) -> dict[str, Path]:
    names = ("yunet", "sface", "pphumanseg") if body else ("yunet", "sface")
    return {name: store.resolve(str(Path("imageprocessing") / "models" / REVISION
                                   / MODEL_SPECS[name]["filename"])) for name in names}


def download(url: str, destination: Path, limit: int, checksum: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ailias/0.1"})
        with os.fdopen(handle, "wb") as target, urllib.request.urlopen(request, timeout=60) as source:
            size = 0
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValueError("Download exceeded pinned size limit")
                target.write(chunk)
        if size == 0 or (checksum and digest(Path(temporary)) != checksum):
            raise ValueError("Downloaded model failed SHA256 verification")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install_models(store: Store, body: bool = True) -> dict:
    paths = model_paths(store, body)
    records = {}
    for name, path in paths.items():
        spec = MODEL_SPECS[name]
        relative = f"models/{spec['directory']}/{spec['filename']}"
        url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/{relative}"
        if not path.is_file() or digest(path) != spec["sha256"]:
            download(url, path, spec["size"], spec["sha256"])
        license_path = path.with_suffix(".LICENSE.txt")
        license_url = (f"https://raw.githubusercontent.com/opencv/opencv_zoo/"
                       f"{REVISION}/models/{spec['directory']}/LICENSE")
        if not license_path.is_file():
            download(license_url, license_path, 1024 * 1024)
        records[name] = {**spec, "source": url, "license_source": license_url,
                         "license_sha256": digest(license_path),
                         "path": str(path.relative_to(store.root))}
    result = {"provider": "opencv/opencv_zoo", "revision": REVISION,
              "device": "cpu", "dtype": "float32", "models": records}
    atomic_json(next(iter(paths.values())).parent / "provenance.json", result)
    return result


def verified_models(store: Store, body: bool) -> dict[str, Path]:
    paths = model_paths(store, body)
    for name, path in paths.items():
        if not path.is_file() or digest(path) != MODEL_SPECS[name]["sha256"]:
            raise ValueError(f"Missing/unverified {name} model; run imageprocessing models first")
    return paths

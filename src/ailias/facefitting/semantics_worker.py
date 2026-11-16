"""Prepare photo/semantic evidence in an owned OpenCV CUDA process, separate from Torch."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ailias.imageprocessing.faceparsing import DIRECTORY, HeadParser
from ailias.storage import Store, atomic_json


def main(request_path):
    request = json.loads(request_path.read_text(encoding="utf-8"))
    store = Store(Path(request["data_root"]), request["alias"])
    root = Path(request["output"])
    parser = HeadParser(store, "cuda", int(request["device"].split(":")[1]))
    result = []
    try:
        for index, item in enumerate(request["photos"]):
            entry = item["entry"]
            if not all(store.verify(entry[name]) for name in ("image", "foreground_alpha", "source")):
                raise ValueError("Texture photo evidence changed")
            with Image.open(store.artifact_path(entry["image"])) as opened:
                image = opened.convert("RGB")
            with Image.open(store.artifact_path(entry["foreground_alpha"])) as opened:
                alpha = opened.convert("L")
            x, y, width, height = entry["box_xywh"]
            scale = min(1, request["photo_size"]/max(image.size))
            original_size = image.size
            if scale < 1:
                size = tuple(round(v*scale) for v in image.size)
                image = image.resize(size, Image.Resampling.LANCZOS)
                alpha = alpha.resize(size, Image.Resampling.BILINEAR)
            actual_scale = np.asarray(image.size)/original_size
            bounds = [x*actual_scale[0], y*actual_scale[1],
                      width*actual_scale[0], height*actual_scale[1]]
            labels = np.asarray(parser.parse(image, bounds))
            rgb, opacity = np.asarray(image), np.asarray(alpha)
            skin = (labels == 1) & (opacity >= 250)
            hair = (labels == 17) & (opacity >= 250)
            if skin.sum() < 100:
                raise ValueError("Insufficient supported skin in a selected texture view")
            pixels = rgb[skin].astype(np.float32)/255
            skin_color = np.median(pixels, axis=0)
            hair_color = np.median(rgb[hair], axis=0)/255 if hair.sum() >= 30 else None
            path = root / f"photo-{index:02}.npz"
            np.savez_compressed(path, rgb=rgb, alpha=opacity, labels=labels)
            result.append({
                "face_id": entry["face_id"], "data": str(path),
                "size": list(image.size), "portrait_scale": actual_scale.tolist(),
                "skin_color": skin_color.tolist(),
                "hair_color": hair_color.tolist() if hair_color is not None else None,
                "hair_top_pixel": float(np.quantile(np.nonzero(hair)[0], .02)) if hair.any() else None,
            })
    finally:
        provenance = parser.provenance
        provenance["license_artifact"] = json.loads(
            store.resolve(str(Path(DIRECTORY) / "provenance.json")).read_text())["license_artifact"]
        parser.close()
    atomic_json(root / "semantics.json", {"photos": result, "parser": provenance})


if __name__ == "__main__":
    main(Path(sys.argv[1]))

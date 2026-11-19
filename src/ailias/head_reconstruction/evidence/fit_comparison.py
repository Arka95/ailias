"""Publish source-pixel landmark reprojection overlays for every fitted image."""

import numpy as np
from PIL import Image, ImageDraw

from ailias.imageprocessing.image_io import read_oriented_image as read_image

from ailias.storage import atomic_json


def write_fit_overlays(store, root, views, fitted, evidence):
    directory = root / "fit-overlays"
    directory.mkdir()
    frames = {frame["face_id"]: frame for frame in evidence["frames"]}
    metrics = {view["face_id"]: view for view in fitted["report"]["views"]}
    artifacts, index = {}, []
    for number, view in enumerate(views):
        frame = frames[view["face_id"]]
        size = np.asarray(view["source_size"])
        observed = view["target"]*max(size)+size/2
        projected = fitted["projected"][number]*max(size)+size/2
        active = view["weights"] > 0
        original, _ = read_image(store.artifact_path(frame["image"]))
        bounds = frame.get("reviewed_crop", [0, 0, *original.size])
        image = original.crop(bounds)
        original.close()
        try:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            left, top, right, bottom = bounds
            scale = np.asarray(image.size)/[right-left, bottom-top]
            draw = ImageDraw.Draw(image)
            for target, predicted in zip((observed[active]-[left, top])*scale,
                                         (projected[active]-[left, top])*scale):
                draw.line([tuple(target), tuple(predicted)], fill="#ffd34e", width=1)
                for point, color in ((target, "#36ff6d"), (predicted, "#ff42d7")):
                    x, y = point
                    draw.ellipse((x-1, y-1, x+1, y+1), fill=color)
            draw.rectangle((0, 0, image.width, 22), fill="black")
            draw.text((5, 5), "Green: detected / Magenta: fitted / Yellow: residual", fill="white")
            path = directory / f"view-{number:03d}.jpg"
            image.save(path, quality=92)
        finally:
            image.close()
        reference = store.reference_artifact(path)
        artifacts[f"fit_overlay_{number}"] = reference
        index.append({"face_id": view["face_id"], "image": reference, **metrics[view["face_id"]]})
    path = directory / "manifest.json"
    atomic_json(path, {"views": index, "landmarks_are_predictions": True,
                       "camera_coordinate_frame": "unchanged input image pixels"})
    artifacts["fit_overlay_index"] = store.reference_artifact(path)
    return artifacts

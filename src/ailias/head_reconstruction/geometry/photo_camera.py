"""Prepare original-photo contour tracking with an existing fitted camera, without refitting."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from ailias.common.identifiers import validate_identifier as component
from ailias.imageprocessing.image_io import read_oriented_image as read_image
from ailias.storage import atomic_json


def perspective_camera(view, center, image_size):
    """Convert existing world-to-CV extrinsics to the GNM camera frame consumed by UE."""
    camera, center = np.asarray(view["camera"], dtype=float), np.asarray(center, dtype=float)
    if (camera.shape != (7,) or center.shape != (3,)
            or not np.isfinite(camera).all() or not np.isfinite(center).all()
            or not -6 < camera[6] < 6):
        raise ValueError("Photo tracking requires finite fitted perspective camera parameters")
    source_size = np.asarray(view["source_size"], dtype=float)
    output_size = np.asarray(image_size, dtype=float)
    if (source_size.shape != (2,) or output_size.shape != (2,)
            or not np.isfinite(source_size).all() or not np.isfinite(output_size).all()
            or np.any(source_size <= 0) or np.any(output_size <= 0)
            or abs(output_size[0]/output_size[1]-source_size[0]/source_size[1]) > .005):
        raise ValueError("Tracking image must retain the fitted source-image aspect ratio")
    rotation = Rotation.from_rotvec(camera[:3]).as_matrix()
    focal = math.exp(camera[6])*max(source_size)
    return {
        "projection": "perspective", "image_size": list(map(int, image_size)),
        "position_gnm": (center-rotation.T@camera[3:6]).tolist(),
        "forward_gnm": (rotation.T@np.array([0., 0., 1.])).tolist(),
        "up_gnm": (rotation.T@np.array([0., -1., 0.])).tolist(),
        "horizontal_fov_degrees": math.degrees(2*math.atan(source_size[0]/(2*focal))),
        "source_size": view["source_size"],
        "camera_mode": "existing_fitted_photo_camera_no_refit",
    }


def prepare_photo_tracking(store, fit_job, face_id, output):
    """Mask other people/background while preserving original full-frame perspective."""
    fitted_path = store.resolve(str(Path("facefitting") / "runs" / component(fit_job) / "manifest.json"))
    dependencies = [store.reference_artifact(fitted_path)]
    fitted = json.loads(fitted_path.read_text(encoding="utf-8-sig"))
    references = [fitted["inputs"], fitted["artifacts"]["cameras"], fitted["artifacts"]["observations"]]
    dependencies.extend(references)
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Photo tracking fit evidence changed")
    inputs, cameras, observations = [
        json.loads(store.artifact_path(item).read_text(encoding="utf-8-sig")) for item in references]
    entries = [item for item in inputs["inputs"]["entries"] if item["face_id"] == face_id]
    views = [item for item in cameras["views"] if item["face_id"] == face_id]
    observed = [item for item in observations["faces"] if item["face_id"] == face_id]
    if len(entries) != 1 or len(views) != 1 or len(observed) != 1 or observed[0]["status"] != "ready":
        raise ValueError("Photo tracking requires one existing fitted and observed face ID")
    entry, view, observation = entries[0], views[0], observed[0]
    if (entry["source_id"] != entry["source"]["sha256"]
            or entry["source_id"] != observation["source_id"]):
        raise ValueError("Photo tracking source identity differs from dense evidence")
    dependencies.extend(entry[key] for key in ("source", "foreground_alpha"))
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Photo tracking source evidence changed")
    image, _ = read_image(store.artifact_path(entry["source"]))
    if list(image.size) != view["source_size"] or list(image.size) != observation["source_size"]:
        raise ValueError("Photo tracking dimensions differ from the fitted source camera")
    left, top, right, bottom = entry["portrait_bounds"]
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError("Photo tracking portrait bounds lie outside the source image")
    with Image.open(store.artifact_path(entry["foreground_alpha"])) as value:
        alpha = value.convert("L")
    if alpha.size != (right-left, bottom-top) or alpha.getbbox() is None:
        raise ValueError("Photo tracking requires a nonempty native-grid portrait alpha")
    tracking = Image.new("RGB", image.size, (128, 128, 128))
    tracking.paste(image.crop((left, top, right, bottom)), (left, top), alpha)
    tracking.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    camera = perspective_camera(view, cameras["model_center"], tracking.size)
    camera.update(
        reference_face_id=face_id, source=entry["source"],
        expression=observation.get("blendshapes", {}), expression_matched=False,
        photo_pixels="original decoded RGB; existing head alpha over neutral full-frame background",
        image_warp="aspect-preserving resize only; no eye alignment, crop or face deformation",
    )
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Photo tracking source evidence changed during preparation")
    output.mkdir(parents=True, exist_ok=True)
    tracking.save(output / "tracking-view.png")
    atomic_json(output / "tracking-camera.json", camera)
    return {"mode": "original_photo_contours", "face_id": face_id,
            "input_dependencies": dependencies, "camera": camera,
            "geometry_refitted": False, "blender_launched": False}

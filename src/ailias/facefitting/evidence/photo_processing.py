"""Recompute face and appearance evidence from immutable, already sampled photos."""

from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
from pathlib import Path

import numpy as np
from ailias.imageprocessing.appearance_features import photo_features
from ailias.imageprocessing.pipeline import read_image
from ailias.imageprocessing.dense import DenseDetector
from ailias.imageprocessing.faceparsing import HeadParser
from ailias.imageprocessing.hair_evidence import facial_hair_evidence
from ailias.facefitting.evidence.frame_analysis import analyze_face_frame
from ailias.storage import atomic_json


CAPTURE_FIELDS = (
    "face_id", "source_id", "source_name", "image", "original_source_size",
    "decode_index", "pts", "time_base", "timestamp_seconds",
    "container_time_seconds", "video_start_seconds",
)


def load_retained_frames(store, manifest):
    """Read only retained image references and capture metadata, never old predictions."""
    reference = store.reference_artifact(manifest)
    evidence = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if evidence.get("status") != "video_evidence_ready":
        raise ValueError("Require a completed sampled-frame manifest")
    frames = evidence.get("frames", [])
    identifiers = [frame["face_id"] for frame in frames]
    if not frames or len(set(identifiers)) != len(identifiers):
        raise ValueError("Sampled frames require nonempty, unique face IDs")
    samples = [{key: frame[key] for key in CAPTURE_FIELDS if key in frame} for frame in frames]
    for frame in samples:
        if not store.verify(frame["image"]):
            raise ValueError("Retained sampled image is missing or changed: " + frame["face_id"])
    return samples, [reference, *(frame["image"] for frame in samples)]


def decode_frame_image(store, frame):
    try:
        return read_image(store.artifact_path(frame["image"]))[0]
    except (OSError, ValueError) as error:
        raise ValueError("Cannot decode sampled image: " + frame["face_id"]) from error


def select_reviewed_face(detection, target):
    """Use only reviewed crop localization to disambiguate freshly predicted faces."""
    if target is None or not detection["faces"]:
        return detection
    x, y, width, height = target
    low, high = np.array([x, y]), np.array([x+width, y+height])
    overlaps = []
    for face in detection["faces"]:
        points = np.asarray(face)[:468, :2]
        minimum, maximum = points.min(0), points.max(0)
        intersection = np.prod(np.maximum(0, np.minimum(high, maximum)-np.maximum(low, minimum)))
        union = width*height+np.prod(maximum-minimum)-intersection
        overlaps.append(intersection/max(union, 1e-12))
    selected = int(np.argmax(overlaps))
    if overlaps[selected] <= 0 or sum(np.isclose(overlaps, overlaps[selected])) != 1:
        return {key: [] for key in ("faces", "blendshapes", "transforms")}
    return {key: [detection[key][selected]] if detection[key] else []
            for key in ("faces", "blendshapes", "transforms")}


def detect_frame_faces(image, frame, detector, policy):
    bounds = frame.get("detection_bounds", [0, 0, *image.size])
    left, top, right, bottom = bounds
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError("Face detection region lies outside the original image")
    with image.crop(bounds) as region:
        region.thumbnail((policy.detection_long_side, policy.detection_long_side))
        detection = detector.detect(np.asarray(region))
    if frame.get("detection_bounds") is not None:
        for face in detection["faces"]:
            points = np.asarray(face, dtype=float)
            points[:, :2] = (points[:, :2]*[right-left, bottom-top]+[left, top])/image.size
            face[:] = points.tolist()
    return select_reviewed_face(detection, frame.get("target_face_box"))


def extract_photo_evidence(store, samples, root, landmarker, policy, *, detector=None, parser=None):
    """Bound CPU image prefetch; serialize each owned inference model and release it on failure."""
    output = root / "photo-evidence"
    output.mkdir(exist_ok=False)
    report = {
        "status": "running", "frames": [], "faces": {}, "artifacts": {},
        "sampled_images_reused": True, "previous_predictions_reused": False,
        "video_decoding_performed": False, "decode_workers": 2, "decode_batch_size": 4,
        "observed_colours_are_intrinsic_albedo": False,
    }
    manifest = output / "manifest.json"
    owns_detector, owns_parser = detector is None, parser is None
    try:
        if detector is None:
            detector = DenseDetector(store.artifact_path(landmarker))
        if parser is None:
            parser = HeadParser(store, "cpu")
        report["parser"] = parser.provenance
        report["landmarker"] = landmarker
        remaining = iter(samples)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="sample-decode") as executor:
            while batch := list(islice(remaining, 4)):
                images = executor.map(lambda frame: decode_frame_image(store, frame), batch)
                for frame, image in zip(batch, images):
                    try:
                        detection = detect_frame_faces(image, frame, detector, policy)
                        observation = analyze_face_frame(np.asarray(image), detection, policy)
                        fresh = {**frame, **observation}
                        report["frames"].append(fresh)
                        if "source_xy" not in observation:
                            report["faces"][frame["face_id"]] = {
                                "status": "not_observed", "reasons": observation["reasons"]}
                            continue
                        left, top, right, bottom = observation["face_box"]
                        parts = parser.parse(image, [left, top, right-left, bottom-top])
                        features = photo_features(
                            np.asarray(image), np.asarray(observation["source_xy"]),
                            parts, np.asarray(observation["on_head"], dtype=bool))
                        skin = features["regions"]["skin"]
                        if skin["status"] == "observed":
                            features["facial_hair_appearance"] = facial_hair_evidence(
                                np.asarray(image), observation["source_xy"], np.asarray(parts),
                                skin["median_linear_rgb"])
                        regions = output / (frame["face_id"] + "-regions.png")
                        parts.save(regions)
                        reference = store.reference_artifact(regions)
                        report["faces"][frame["face_id"]] = {
                            "status": "observed", "source": frame["image"],
                            "source_size": list(image.size), "features": features, "regions": reference,
                            "region_coordinate_frame": "unchanged sampled-image pixels",
                        }
                        report["artifacts"][frame["face_id"]] = reference
                    except (OSError, ValueError, RuntimeError) as error:
                        raise RuntimeError("Photo processing failed: " + frame["face_id"]) from error
                    finally:
                        image.close()
                atomic_json(manifest, report)
        report["status"] = "photo_evidence_ready"
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        for resource, owned in ((parser, owns_parser), (detector, owns_detector)):
            if resource is not None and owned:
                try:
                    resource.close()
                except Exception as error:
                    report.setdefault("cleanup_errors", []).append(type(error).__name__)
        if report.get("cleanup_errors") and report["status"] != "failed":
            report["status"] = "failed"
        atomic_json(manifest, report)
    if report["status"] != "photo_evidence_ready":
        raise RuntimeError("Photo processing resource cleanup failed; inspect " + str(manifest))
    return report

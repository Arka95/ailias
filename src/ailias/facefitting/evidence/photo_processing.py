"""Recompute face and appearance evidence from immutable, already sampled photos."""

from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ailias.imageprocessing.appearance_features import photo_features
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
        with Image.open(store.artifact_path(frame["image"])) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError("Cannot decode sampled image: " + frame["face_id"]) from error


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
                        small = image.copy()
                        small.thumbnail((policy.detection_long_side, policy.detection_long_side))
                        observation = analyze_face_frame(
                            np.asarray(image), detector.detect(np.asarray(small)), policy)
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

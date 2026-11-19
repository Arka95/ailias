"""Prepare every accepted original-photo portrait across groups, without reusing predictions."""

import json
from pathlib import Path

from ailias.common.identifiers import validate_identifier as component
from ailias.storage import atomic_json


CAPTURE_FIELDS = (
    "face_id",
    "source_id",
    "source_name",
    "image",
    "original_source_size",
    "decode_index",
    "pts",
    "time_base",
    "timestamp_seconds",
    "container_time_seconds",
    "video_start_seconds",
)


def load_sampled_photo_inputs(store, manifest):
    """Load verified retained images and capture metadata without reusing predictions or video decoding.
    Treat existing sampled-frame manifests as an input format, not a feature-extraction implementation.
    """
    reference = store.reference_artifact(manifest)
    evidence = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if evidence.get("status") != "video_evidence_ready":
        raise ValueError("Require a completed sampled-frame manifest")
    frames = evidence.get("frames", [])
    identifiers = [component(frame["face_id"]) for frame in frames]
    if not frames or len(set(identifiers)) != len(identifiers):
        raise ValueError("Sampled frames require nonempty, unique face IDs")
    samples = [
        {key: frame[key] for key in CAPTURE_FIELDS if key in frame} for frame in frames
    ]
    for frame in samples:
        if not store.verify(frame["image"]):
            raise ValueError(
                "Retained sampled image is missing or changed: " + frame["face_id"]
            )
    return samples, [reference, *(frame["image"] for frame in samples)]


def prepare_reviewed_photos(store, run, root):
    """Load explicitly accepted original-photo crops and retain their original file provenance.
    Leave pixel preparation and all fresh model predictions to imageprocessing.
    """
    source_manifest = store.resolve(
        str(Path("imageprocessing") / "runs" / component(run) / "manifest.json")
    )
    reference = store.reference_artifact(source_manifest)
    scan = json.loads(source_manifest.read_text(encoding="utf-8-sig"))
    if scan.get("status") != "complete":
        raise ValueError("Original-photo intake must be complete")
    directory = root / "photo-inputs"
    directory.mkdir(exist_ok=False)
    report = {
        "run": run,
        "source_manifest": reference,
        "status": "running",
        "sources": {},
        "selected": [],
        "excluded_faces": [],
        "previous_predictions_reused": False,
    }
    samples, dependencies = [], [reference]
    try:
        for source_id, source in scan["sources"].items():
            faces = [scan["faces"][face_id] for face_id in source["faces"]]
            accepted = [
                face
                for face in faces
                if face.get("curation", {}).get("decision") == "accept"
            ]
            report["sources"][source_id] = {
                "source": source["artifact"],
                "accepted_faces": [face["id"] for face in accepted],
                "status": "selected" if accepted else "no_accepted_target_portrait",
            }
            report["excluded_faces"].extend(
                {
                    "face_id": face["id"],
                    "reason": face.get("curation", {}).get("decision", "not_reviewed"),
                }
                for face in faces
                if face not in accepted
            )
            if not accepted:
                continue
            artifact = source["artifact"]
            if not store.verify(artifact):
                raise ValueError("Original photo changed after review: " + source_id)
            source_width, source_height = source["metadata"]["oriented_size"]
            for face in accepted:
                bounds = face["portrait_bounds"]
                left, top, right, bottom = bounds
                if not (
                    0 <= left < right <= source_width
                    and 0 <= top < bottom <= source_height
                ):
                    raise ValueError(
                        "Reviewed crop lies outside its original photo: " + face["id"]
                    )
                sample = {
                    "face_id": "photo-" + face["id"],
                    "source_id": source_id,
                    "source_name": store.artifact_path(artifact).name,
                    "image": artifact,
                    "original_photo": artifact,
                    "original_source_size": [source_width, source_height],
                    "reviewed_crop": bounds,
                    "capture_kind": "original_photo",
                }
                samples.append(sample)
                report["selected"].append(sample)
            dependencies.append(artifact)
        if not samples:
            raise ValueError(
                "Original-photo intake contains no accepted target portraits"
            )
        report.update(
            status="reviewed_photos_ready",
            source_count=len(scan["sources"]),
            selected_portraits=len(samples),
        )
        return samples, dependencies
    except Exception as error:
        report.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        atomic_json(directory / "manifest.json", report)

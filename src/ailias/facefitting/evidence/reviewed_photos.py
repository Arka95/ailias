"""Prepare every accepted original-photo portrait across groups, without reusing predictions."""

import json
from pathlib import Path

from ailias.imageprocessing.pipeline import component
from ailias.storage import atomic_json


def prepare_reviewed_photos(store, run, root):
    source_manifest = store.resolve(str(Path("imageprocessing") / "runs" / component(run) / "manifest.json"))
    reference = store.reference_artifact(source_manifest)
    scan = json.loads(source_manifest.read_text(encoding="utf-8-sig"))
    if scan.get("status") != "complete":
        raise ValueError("Original-photo intake must be complete")
    directory = root / "photo-inputs"
    directory.mkdir(exist_ok=False)
    report = {"run": run, "source_manifest": reference, "status": "running", "sources": {},
              "selected": [], "excluded_faces": [], "previous_predictions_reused": False}
    samples, dependencies = [], [reference]
    try:
        for source_id, source in scan["sources"].items():
            faces = [scan["faces"][face_id] for face_id in source["faces"]]
            accepted = [face for face in faces if face.get("curation", {}).get("decision") == "accept"]
            report["sources"][source_id] = {
                "source": source["artifact"], "accepted_faces": [face["id"] for face in accepted],
                "status": "selected" if accepted else "no_accepted_target_portrait",
            }
            report["excluded_faces"].extend({
                "face_id": face["id"], "reason": face.get("curation", {}).get("decision", "not_reviewed")}
                for face in faces if face not in accepted)
            if not accepted:
                continue
            artifact = source["artifact"]
            if not store.verify(artifact):
                raise ValueError("Original photo changed after review: " + source_id)
            source_width, source_height = source["metadata"]["oriented_size"]
            for face in accepted:
                bounds = face["portrait_bounds"]
                left, top, right, bottom = bounds
                if not (0 <= left < right <= source_width and 0 <= top < bottom <= source_height):
                    raise ValueError("Reviewed crop lies outside its original photo: " + face["id"])
                sample = {"face_id": "photo-" + face["id"], "source_id": source_id,
                          "source_name": store.artifact_path(artifact).name,
                          "image": artifact, "original_photo": artifact,
                          "original_source_size": [source_width, source_height], "reviewed_crop": bounds,
                          "detection_bounds": bounds, "capture_kind": "original_photo"}
                x, y, width, height = face["box_xywh"]
                sample["target_face_box"] = [
                    x/source_width, y/source_height, width/source_width, height/source_height]
                samples.append(sample)
                report["selected"].append(sample)
            dependencies.append(artifact)
        if not samples:
            raise ValueError("Original-photo intake contains no accepted target portraits")
        report.update(status="reviewed_photos_ready", source_count=len(scan["sources"]),
                      selected_portraits=len(samples))
        return samples, dependencies
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        atomic_json(directory / "manifest.json", report)

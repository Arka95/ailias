"""Extract reusable per-photo features from explicitly reused dense observations."""

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ailias.imageprocessing.appearance_features import photo_features
from ailias.imageprocessing.curation import PROFILE
from ailias.imageprocessing.image_io import read_oriented_image as read_image
from ailias.storage import atomic_json


def extract_features(store, inputs_path, observations_path, cohort, parser, face_ids=None):
    """Persist generic evidence; never refit identity, mutate photos or select native assets."""
    if not isinstance(cohort, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}", cohort):
        raise ValueError("Feature cohort requires a 1-80 character descriptive label")
    inputs_ref = store.reference_artifact(inputs_path)
    observations_ref = store.reference_artifact(observations_path)
    prepared = json.loads(inputs_path.read_text(encoding="utf-8-sig"))
    dense = json.loads(observations_path.read_text(encoding="utf-8-sig"))
    if prepared.get("inputs", {}).get("schema_version") != 1:
        raise ValueError("Require version 1 prepared photo inputs")
    if dense.get("method") != "mediapipe-478-reversible-source-v1":
        raise ValueError("Require reversible 478-point photo observations")
    entries = prepared["inputs"]["entries"]
    faces = dense["faces"]
    if (not entries or len({entry["face_id"] for entry in entries}) != len(entries)
            or len({face["face_id"] for face in faces}) != len(faces)):
        raise ValueError("Photo inputs and observations require unique face IDs")
    by_id = {entry["face_id"]: entry for entry in entries}
    observed = {face["face_id"]: face for face in faces}
    selected = sorted(by_id if face_ids is None else set(face_ids))
    if not selected or set(selected) - by_id.keys() or set(selected) - observed.keys():
        raise ValueError("Selected face IDs must exist in both inputs and observations")
    curation = store.reference_artifact(store.resolve(PROFILE))
    if prepared["inputs"].get("curation_sha256") != curation["sha256"]:
        raise ValueError("Prepared photo inputs are stale for current curation")
    dependencies = [inputs_ref, observations_ref, curation]
    for key in ("model", "license_artifact"):
        if key in parser.provenance:
            dependencies.append(parser.provenance[key])
    for face_id in selected:
        entry, observation = by_id[face_id], observed[face_id]
        if (entry["source_id"] != observation["source_id"]
                or entry["source_id"] != entry["source"]["sha256"]):
            raise ValueError("Observation source differs from prepared photo input")
        dependencies.extend(entry[key] for key in ("source", "portrait"))
        if observation.get("status") == "ready":
            analysis = entry["analysis"]
            if (observation["alignment"] != analysis["alignment"]
                    or observation["aligned_cutout"] != analysis["aligned_cutout"]):
                raise ValueError("Observation coordinates differ from the prepared photo")
            dependencies.append(analysis["aligned_cutout"])
    if not all(store.verify(reference) for reference in dependencies):
        raise ValueError("Feature source evidence changed before extraction")
    revision = uuid.uuid4().hex
    root = store.resolve(str(Path("imageprocessing") / "features" / revision))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    report = {
        "schema_version": 1, "revision": revision, "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "photo-regions-and-projected-landmarks-v1", "cohort": cohort,
        "status": "running", "run": prepared.get("run"), "group": prepared.get("group"),
        "input_dependencies": dependencies, "observation_mode": "explicit_reuse_no_dense_inference",
        "parser": parser.provenance, "faces": {}, "artifacts": {},
        "identity_confirmed": False, "native_edits_applied": False,
        "limitations": [
            "No cross-photo appearance averaging; the cohort label does not attest matching capture",
            "Photo-space ratios depend on pose/expression/camera and are not calibrated 3D dimensions",
            "Observed colours retain photographed lighting and are not intrinsic albedo",
            "Beard, moustache and eyelashes require separate evidence, not the scalp-hair class",
        ],
    }
    atomic_json(manifest, report)
    try:
        for index, face_id in enumerate(selected):
            entry, observation = by_id[face_id], observed[face_id]
            record = {
                "source_id": entry["source_id"], "source": entry["source"],
                "cohort": cohort, "view": entry.get("view", {}),
                "quality_warnings": entry.get("quality_warnings", []),
                "expression": observation.get("blendshapes", {}),
                "facial_transform": observation.get("facial_transform"),
            }
            if observation["status"] != "ready":
                report["faces"][face_id] = {
                    **record, "status": "review", "reason": observation.get("reason", "dense_not_ready")}
                continue
            image, metadata = read_image(store.artifact_path(entry["source"]))
            if metadata["oriented_size"] != observation["source_size"]:
                raise ValueError("Photo dimensions differ from dense observation coordinates")
            points = np.asarray(observation["source_xy"], dtype=float)
            aligned = np.asarray(observation["aligned_xy"], dtype=float)
            transform = np.asarray(observation["alignment"]["aligned_to_source"], dtype=float)
            inverse = np.asarray(observation["alignment"]["source_to_aligned"], dtype=float)
            if (points.shape != (478, 2) or aligned.shape != (478, 2)
                    or transform.shape != (3, 3) or inverse.shape != (3, 3)
                    or not np.isfinite(points).all() or not np.isfinite(aligned).all()
                    or not np.isfinite(transform).all() or not np.isfinite(inverse).all()
                    or not np.allclose(transform @ inverse, np.eye(3), atol=1e-6)
                    or not np.allclose((np.c_[aligned, np.ones(478)] @ transform.T)[:, :2],
                                       points, atol=.01, rtol=0)):
                raise ValueError("Dense landmarks do not round-trip to source coordinates")
            left, top, right, bottom = entry["portrait_bounds"]
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError("Prepared portrait lies outside the original photo")
            portrait = image.crop((left, top, right, bottom))
            parts = parser.parse(portrait, entry["box_xywh"])
            evidence = photo_features(np.asarray(portrait), points-[left, top], parts,
                                      observation["on_head"])
            regions_path = root / f"regions-{index:03d}.png"
            parts.save(regions_path)
            regions_ref = store.reference_artifact(regions_path)
            report["artifacts"][f"regions_{index}"] = regions_ref
            report["faces"][face_id] = {
                **record, "status": "observed", "features": evidence,
                "source_size": observation["source_size"], "source_xy": points.tolist(),
                "portrait_origin_xy": [left, top], "regions": regions_ref,
                "region_coordinate_frame": "portrait pixels; add portrait_origin_xy for source pixels",
                "source_decode": metadata,
            }
            atomic_json(manifest, report)
        if not all(store.verify(reference) for reference in dependencies):
            raise ValueError("Feature source evidence changed during extraction")
        report["summary"] = dict(Counter(face["status"] for face in report["faces"].values()))
        report["status"] = "features_extracted_awaiting_review"
        report["manifest"] = str(manifest)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        atomic_json(manifest, report)
    return report


def inspect_image(store, image_path, cohort, parser, detector, model_reference, role):
    """Measure one explicit photo/render without fitting geometry or requiring photo curation."""
    if not isinstance(cohort, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}", cohort):
        raise ValueError("Feature cohort requires a 1-80 character descriptive label")
    if role not in {"reference", "render"}:
        raise ValueError("Image role must be reference or render")
    source = store.reference_artifact(image_path)
    dependencies = [source, model_reference]
    dependencies.extend(parser.provenance[key] for key in ("model", "license_artifact")
                        if key in parser.provenance)
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Image inspection inputs changed")
    image, metadata = read_image(image_path)
    result = detector.detect(np.asarray(image))
    revision = uuid.uuid4().hex
    root = store.resolve(str(Path("imageprocessing") / "features" / revision))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    report = {
        "schema_version": 1, "revision": revision, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "method": "photo-regions-and-projected-landmarks-v1",
        "cohort": cohort, "image_role": role, "input_dependencies": dependencies,
        "observation_mode": "fresh_explicit_cpu_dense_inference",
        "landmarker": {"provider": "MediaPipe Face Landmarker", "model": model_reference, "device": "cpu"},
        "parser": parser.provenance, "faces": {}, "artifacts": {},
        "native_edits_applied": False, "identity_confirmed": False,
        "image_role_is_caller_declared": True,
    }
    atomic_json(manifest, report)
    try:
        if len(result["faces"]) != 1:
            report.update(status="review", reason="require_exactly_one_detected_face",
                          detected_faces=len(result["faces"]))
        else:
            predicted = np.asarray(result["faces"][0], dtype=float)
            if predicted.shape != (478, 3) or not np.isfinite(predicted).all():
                raise ValueError("Image inspection requires 478 finite predicted landmarks")
            points = predicted[:, :2]*image.size
            minimum, maximum = points.min(0), points.max(0)
            bounds = [*minimum.tolist(), *(maximum-minimum).tolist()]
            parts = parser.parse(image, bounds)
            supported = ((points >= 0) & (points < image.size)).all(1)
            evidence = photo_features(np.asarray(image), points, parts, supported)
            regions = root / "regions-000.png"
            parts.save(regions)
            artifact = store.reference_artifact(regions)
            report["artifacts"]["regions_0"] = artifact
            face_id = source["sha256"] + "-00"
            report["faces"][face_id] = {
                "status": "observed", "source_id": source["sha256"], "source": source,
                "source_size": list(image.size), "source_xy": points.tolist(),
                "features": evidence, "cohort": cohort, "regions": artifact,
                "portrait_origin_xy": [0, 0], "region_coordinate_frame": "source image pixels",
                "expression": result.get("blendshapes", [{}])[0] if result.get("blendshapes") else {},
                "facial_transform": result["transforms"][0] if result.get("transforms") else None,
                "source_decode": metadata,
            }
            report.update(status="features_extracted_awaiting_review", summary={"observed": 1})
        if not all(store.verify(item) for item in dependencies):
            raise ValueError("Image inspection inputs changed during extraction")
        report["manifest"] = str(manifest)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        atomic_json(manifest, report)
    return report

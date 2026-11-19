"""Scan external photos and publish reusable crops, candidate groups and selections.

Preserves source references and applies reviewed curation without confirming identity.
"""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from ailias.common.identifiers import generate_run_id as new_id, validate_identifier as component
from ailias.imageprocessing.backend import Backend
from ailias.imageprocessing.assets import encode_png, store_png, store_png_bytes
from ailias.imageprocessing.curation import CurationError, load_filter
from ailias.imageprocessing.features import (
    GROUPING_METHOD, contact_sheet, face_quality, group_faces, normalize_embedding,
    overlaps_other, portrait_box, subject_bodies,
)
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.imageprocessing.image_io import read_oriented_image as read_image
from ailias.resource_management import finalize_job_resources
from ailias.resources import cleanup_owned_files
from ailias.storage import Store, atomic_json

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic", ".heif", ".hif"}
def scan(store: Store, source: Path, policy: ProcessingPolicy, backend: Backend,
         recursive: bool = False, body: bool = True, limit: int | None = None,
         progress=None) -> dict:
    run_id = new_id()
    relative = Path("imageprocessing") / "runs" / run_id
    manifest_path = relative / "manifest.json"
    created_assets = []
    try:
        result = _scan(store, source, policy, backend, recursive, body, limit,
                       progress, run_id, created_assets)
        return result
    finally:
        manifest = store.resolve(str(manifest_path))
        if manifest.is_file():
            report = json.loads(manifest.read_text(encoding="utf-8"))
            retained = [str(manifest_path)]
            retained.extend(record["artifact"]["path"] for record in report["sources"].values()
                            if record["artifact"].get("storage") != "external")
            for record in report["faces"].values():
                retained.extend(record["products"].values())
            for group in report["groups"]:
                retained.extend(group.get("contact_sheets", []))
            if report.get("overview"):
                retained.append(report["overview"])
            if report["status"] == "running":
                report["status"] = "failed"
            finalize_job_resources(
                store, report, manifest, [store.resolve(path) for path in retained],
                retain_metadata=False, created_assets=created_assets,
                automatic_retention=False,
            )
            if report["status"] == "complete":
                result["cleanup"] = report["cleanup"]
        else:
            cleanup_owned_files(store, str(relative), [])


def _scan(store: Store, source: Path, policy: ProcessingPolicy, backend: Backend,
          recursive: bool, body: bool, limit: int | None, progress, run_id: str,
          created_assets: list[str]) -> dict:
    source = source.resolve(strict=True)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if source.is_file():
        files = [source] if source.suffix.lower() in SUFFIXES else []
    else:
        files = source.rglob("*") if recursive else source.iterdir()
        files = [path for path in files if path.is_file() and not path.is_symlink()
                 and path.suffix.lower() in SUFFIXES
                 and not path.resolve().is_relative_to(store.root.resolve())]
    files = sorted(files, key=lambda path: str(path).lower())
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError("No supported images found in input")
    portrait_filter = load_filter(store, backend.provenance, policy)
    relative = Path("imageprocessing") / "runs" / run_id
    root = store.resolve(str(relative))
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "run": run_id, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(source), "recursive": recursive, "body_enabled": body,
        "source_storage": "reference-only",
        "generated_storage": "alias-content-addressed-assets-v1",
        "policy": asdict(policy), "backend": backend.provenance,
        "grouping_method": GROUPING_METHOD,
        "sources": {}, "faces": {}, "groups": [], "file_errors": [],
        "curation": ({"method": portrait_filter.profile["method"],
                      "profile_sha256": portrait_filter.fingerprint,
                      "reviewed_examples": len(portrait_filter.examples)}
                     if portrait_filter else {"status": "not_configured"}),
        "limitations": [
            "Groups are identity candidates, not confirmed names",
            "No neutral-expression, occlusion, or 3D likeness certification",
            "No generative restoration, denoising, warping, or upscaling",
            "Body masks are coarse visible-person masks, not complete bodies",
        ],
    }
    manifest = root / "manifest.json"
    atomic_json(manifest, report)
    for index, path in enumerate(files):
        try:
            artifact = store.reference_artifact(path)
            source_id = artifact["sha256"]
            if source_id in report["sources"]:
                report["sources"][source_id]["input_files"].append(str(path))
                continue
            image, metadata = read_image(store.artifact_path(artifact))
            if not store.verify(artifact):
                raise ValueError("Source changed during decoding; retry this file")
            record = {"artifact": artifact, "input_files": [str(path)],
                      "metadata": metadata, "faces": []}
            report["sources"][source_id] = record
            detected = backend.detect(image)
            bodies = [{"status": "disabled"} for _ in detected]
            if body and detected:
                try:
                    bodies = subject_bodies(backend.segment(image), detected)
                except (OSError, ValueError) as error:
                    from ailias.imageprocessing.cuda import CudaUnavailable
                    if isinstance(error, CudaUnavailable):
                        raise
                    bodies = [{"status": "segmentation_failed"} for _ in detected]
            for number, (face, subject) in enumerate(zip(detected, bodies)):
                face_id = f"{source_id}-{number:02d}"
                bounds = portrait_box(face.box, image.size, policy)
                quality = face_quality(image, face.box, policy)
                if any(overlaps_other(bounds, other.box) for other in detected if other is not face):
                    quality["reasons"].append("portrait_contains_other_face")
                embedding = normalize_embedding(face.embedding)
                scratch = root / "work"
                decision = (portrait_filter.classify(embedding, quality, source_id, bounds)
                            if portrait_filter else {"decision": "unfiltered"})
                portrait = None
                if decision.get("reason") != "human_reviewed_crop":
                    portrait = encode_png(image.crop(bounds))
                    if portrait_filter:
                        decision = portrait_filter.classify(
                            embedding, quality, source_id, bounds, hashlib.sha256(portrait).hexdigest())
                products = {}
                if decision["decision"] != "reject":
                    if decision["decision"] == "accept" and "reviewed_portrait_sha256" in decision:
                        artifact = portrait_filter.hashes[decision["reviewed_portrait_sha256"]]["artifact"]
                        if not store.verify(artifact):
                            raise CurationError("Reviewed portrait changed during scan; run curate again")
                        products["portrait"] = artifact["path"]
                    else:
                        role = "review" if decision["decision"] == "review" else "portraits"
                        products["portrait"] = store_png_bytes(
                            store, portrait, role, scratch, created_assets)
                body_info = {key: value for key, value in subject.items() if key != "mask"}
                if decision["decision"] == "reject":
                    body_info = {"status": "rejected_portrait"}
                elif subject["status"] == "visible_subject_only":
                    cutout = image.convert("RGBA")
                    cutout.putalpha(subject["mask"])
                    products["body"] = store_png(
                        store, cutout.crop(subject["bounds"]), "bodies", scratch, created_assets)
                    products["mask"] = store_png(
                        store, subject["mask"].crop(subject["bounds"]), "masks", scratch, created_assets)
                report["faces"][face_id] = {
                    "id": face_id, "source_id": source_id,
                    "box_xywh": [round(value, 3) for value in face.box],
                    "portrait_bounds": list(bounds), "score": face.score,
                    "landmarks_5": face.landmarks, "embedding": embedding,
                    "quality": quality, "eligible": not quality["reasons"],
                    "body": body_info, "products": products, "curation": decision,
                    "requires_capture_review": True, "identity_ambiguous": True,
                }
                if decision["decision"] in {"reject", "review"}:
                    report["faces"][face_id]["eligible"] = False
                record["faces"].append(face_id)
            record["status"] = "detected" if detected else "no_face"
        except (OSError, ValueError, Image.DecompressionBombWarning,
                Image.DecompressionBombError) as error:
            from ailias.imageprocessing.cuda import CudaUnavailable
            if isinstance(error, (CudaUnavailable, CurationError)):
                raise
            report["file_errors"].append({"input_file": str(path), "error": type(error).__name__})
        finally:
            atomic_json(manifest, report)
            if progress:
                progress({"processed": index + 1, "total": len(files), "faces": len(report["faces"])})
    report["groups"] = group_faces(
        [face for face in report["faces"].values() if "portrait" in face["products"]], policy)
    for group in report["groups"]:
        members = [report["faces"][face_id] for face_id in group["members"]]
        group["contact_sheets"] = []
        for offset in range(0, len(members), 60):
            tiles = []
            for face in members[offset:offset + 60]:
                with Image.open(store.resolve(face["products"]["portrait"])) as image:
                    candidate = face["eligible"] and not face["identity_ambiguous"]
                    label = face["id"][:8] + (" candidate" if candidate else " review")
                    tiles.append((label, ImageOps.contain(image.convert("RGB"), (190, 195))))
            sheet_path = relative / "groups" / group["id"] / f"contact-{offset // 60 + 1:02d}.jpg"
            store.resolve(str(sheet_path)).parent.mkdir(parents=True, exist_ok=True)
            contact_sheet(tiles).save(store.resolve(str(sheet_path)), quality=90)
            group["contact_sheets"].append(str(sheet_path))
    overview = []
    for group in report["groups"]:
        first = report["faces"][group["members"][0]]
        with Image.open(store.resolve(first["products"]["portrait"])) as image:
            label = f"{group['id']} n={len(group['members'])} good={group['eligible']}"
            overview.append((label, ImageOps.contain(image.convert("RGB"), (190, 195))))
    overview_path = relative / "groups.jpg"
    contact_sheet(overview).save(store.resolve(str(overview_path)), quality=90)
    report["overview"] = str(overview_path)
    if hasattr(backend, "execution_report"):
        backend.execution_report()
    report["status"] = "complete"
    reasons = Counter(reason for face in report["faces"].values()
                      for reason in face["quality"]["reasons"])
    report["summary"] = {
        "input_files": len(files), "unique_sources": len(report["sources"]),
        "duplicates": sum(len(record["input_files"]) - 1 for record in report["sources"].values()),
        "no_face": sum(record.get("status") == "no_face" for record in report["sources"].values()),
        "errors": len(report["file_errors"]), "faces": len(report["faces"]),
        "groups": len(report["groups"]),
        "eligible_faces": sum(group["eligible"] for group in report["groups"]),
        "quality_reasons": dict(reasons),
        "body_status": dict(Counter(face["body"]["status"] for face in report["faces"].values())),
        "portrait_filter": dict(Counter(face["curation"]["decision"] for face in report["faces"].values())),
    }
    atomic_json(manifest, report)
    return {"run": run_id, "manifest": str(manifest), "overview": str(store.resolve(str(overview_path))),
            "summary": report["summary"],
            "groups": [{"id": group["id"], "faces": len(group["members"]),
                        "eligible": group["eligible"]} for group in report["groups"]]}


def load_run(store: Store, run: str) -> dict:
    path = store.resolve(str(Path("imageprocessing") / "runs" / component(run) / "manifest.json"))
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("Run is incomplete or has an unsupported schema")
    return report


def select_group(store: Store, run: str, group_id: str) -> dict:
    selection_id = new_id()
    relative = Path("imageprocessing") / "selections" / selection_id
    try:
        return _select_group(store, run, group_id, selection_id)
    finally:
        path = store.resolve(str(relative / "selection.json"))
        retained = []
        if path.is_file():
            selected = json.loads(path.read_text(encoding="utf-8"))
            retained.append(str(relative / "selection.json"))
            retained.extend(artifact["path"] for member in selected["members"]
                            for artifact in member["products"].values()
                            if artifact.get("storage") != "external")
        cleanup_owned_files(store, str(relative), retained)


def _select_group(store: Store, run: str, group_id: str, selection_id: str) -> dict:
    report = load_run(store, run)
    if report.get("grouping_method") != GROUPING_METHOD:
        raise ValueError("This run uses superseded grouping; run scan again before selection")
    group = next((group for group in report["groups"] if group["id"] == component(group_id)), None)
    if group is None:
        raise ValueError("Unknown group")
    faces = [report["faces"][key] for key in group["members"]]
    faces = [face for face in faces if face["eligible"] and not face["identity_ambiguous"]]
    portrait_filter = load_filter(store, report["backend"], ProcessingPolicy(**report["policy"]))
    if portrait_filter:
        faces = [
            face for face in faces if portrait_filter.classify(
                face["embedding"], face["quality"], face["source_id"], face["portrait_bounds"],
            )["decision"] == "accept"
        ]
    if not faces:
        raise ValueError("Group has no eligible unambiguous portrait candidates")
    relative = Path("imageprocessing") / "selections" / selection_id
    selected = []
    for face in faces:
        source = report["sources"][face["source_id"]]["artifact"]
        if not store.verify(source):
            raise ValueError("Source artifact changed since scan")
        products = {"original": source}
        for role, old_path in face["products"].items():
            products[role] = store.reference_artifact(store.resolve(old_path))
        selected.append({"face_id": face["id"], "products": products,
                         "crop_bounds": face["portrait_bounds"],
                         "landmarks_5": face["landmarks_5"], "requires_capture_review": True})
    result = {
        "alias": store.root.name, "run": run, "group": group_id, "selection": selection_id,
        "manifest": str(store.resolve(str(relative / "selection.json"))),
        "selected_count": len(selected), "members": selected,
        "portraits": [str(store.artifact_path(member["products"]["portrait"])) for member in selected],
        "originals": [str(store.artifact_path(member["products"]["original"])) for member in selected],
        "requires_capture_review": True,
    }
    atomic_json(store.resolve(str(relative / "selection.json")), result)
    atomic_json(store.resolve(str(Path("imageprocessing") / "selection.json")), result)
    return result

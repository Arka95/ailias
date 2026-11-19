"""Prepare hash-verified native inputs from human-curated portraits and head masks.

Preserves source provenance and selects supported reconstruction anchors.
"""

import hashlib
import json
import math
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw

from ailias.generation.settings import GenerationPolicy
from ailias.imageprocessing.curation import load_filter
from ailias.imageprocessing.features import contact_sheet
from ailias.common.identifiers import validate_identifier as component
from ailias.imageprocessing.pipeline import load_run
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.imageprocessing.subjects import associate_subject, select_views, view_rank
from ailias.imageprocessing.views import estimate_view
from ailias.resources import cleanup_owned_files
from ailias.storage import Store, atomic_json, digest


def registration_anchors(store, run, group_id, candidates, fingerprint, maximum):
    eligible = {face["id"] for face in candidates}
    models = []
    root = store.resolve(str(Path("generation") / "runs"))
    for path in sorted(root.glob("*\\manifest.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            artifact = job.get("artifacts", {}).get("cameras")
            if (job.get("run") != run or job.get("group") != group_id
                    or job.get("curation", {}).get("sha256") != fingerprint
                    or not artifact or not store.verify(artifact)):
                continue
            cameras = json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))["cameras"]
            ids = sorted({camera["face_id"] for camera in cameras.values()
                          if camera.get("face_id") in eligible})
            if len(ids) >= 3:
                models.append(ids)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not models:
        return []
    # Keep one model's tracked backbone, not every past job or a transitive identity chain.
    return max(models, key=len)[:min(maximum, max(3, maximum//4))]


def prepare_inputs(store: Store, run: str, group_id: str, policy: GenerationPolicy) -> dict:
    report = load_run(store, run)
    group = next((g for g in report["groups"] if g["id"] == component(group_id)), None)
    if group is None:
        raise ValueError("Unknown face group")
    gallery = load_filter(store, report["backend"], ProcessingPolicy(**report["policy"]))
    if gallery is None:
        raise ValueError("Freeze manual portrait review with imageprocessing curate before generation")
    cleansed = None
    if policy.use_cleansed_portraits:
        from ailias.imageprocessing.cleansing import load_cleansing
        cleansed = load_cleansing(store, run, gallery.fingerprint)
    candidates, excluded = [], []
    for face in report["faces"].values():
        if not policy.selection_expand_groups and face["id"] not in group["members"]:
            continue
        relative = face["products"].get("portrait")
        reasons = face["quality"]["reasons"]
        allowed = {"face_too_small", "face_blurred_or_low_detail",
                   "face_underexposed", "face_overexposed"}
        if (relative is None or (reasons and not policy.selection_allow_quality_warnings)
                or set(reasons) - allowed or face["quality"]["native_short_side"] < 64
                and policy.selection_allow_quality_warnings):
            excluded.append({"face_id": face["id"], "reason": "missing_portrait_or_quality"})
            continue
        portrait = store.resolve(relative)
        reviewed = gallery.hashes.get(portrait.stem)
        if reviewed is None or reviewed["label"] != "accept":
            continue
        if relative != reviewed["artifact"]["path"] or not store.verify(reviewed["artifact"]):
            raise ValueError("Reviewed portrait changed; curate again before reconstruction")
        if not store.verify(report["sources"][face["source_id"]]["artifact"]):
            raise ValueError("Original source changed since imageprocessing")
        if cleansed is not None:
            derivative = cleansed["faces"].get(face["id"])
            if derivative is None or derivative["status"] != "ready":
                excluded.append({"face_id": face["id"], "reason": "cleansing_missing_or_requires_review"})
                continue
            if (derivative["portrait"]["path"] != relative
                    or derivative["portrait"]["sha256"] != portrait.stem
                    or derivative["source"] != report["sources"][face["source_id"]]["artifact"]
                    or derivative["portrait_bounds"] != face["portrait_bounds"]):
                raise ValueError("Cleansed portrait no longer matches source/crop metadata")
            for artifact in derivative["products"].values():
                if not store.verify(artifact):
                    raise ValueError("Cleansed portrait/alpha changed; run imageprocessing cleanse again")
        source = report["sources"][face["source_id"]]
        candidates.append({**face, "view": estimate_view(face, source.get("metadata", {})),
                           "lighting": derivative.get("lighting", {}) if cleansed is not None else {}})
    associated, decisions = associate_subject(
        candidates, report["groups"], group_id, ProcessingPolicy(**report["policy"]),
        expand=policy.selection_expand_groups)
    lighting_reference = None
    if cleansed is not None:
        from ailias.imageprocessing.observations import assess_lighting
        lighting_records = {
            face["id"]: {"lighting": deepcopy(face["lighting"]), "subject_scope": group_id,
                         "status": "ready", "view": face["view"],
                         "quality_warnings": face["quality"]["reasons"]}
            for face in associated
        }
        lighting_reference = assess_lighting(lighting_records)
        associated = [dict(face, lighting=lighting_records[face["id"]]["lighting"])
                      for face in associated]
    anchors = registration_anchors(store, run, group_id, associated, gallery.fingerprint, policy.max_images)
    chosen, counts = select_views(associated, policy.max_images, anchors)
    if len(chosen) < policy.min_registered_images:
        raise ValueError("Too few distinct, explicitly kept, quality-eligible photos in this group")
    group_ids = {key: candidate["id"] for candidate in report["groups"] for key in candidate["members"]}
    selection = {
        "method": "seed-anchored-evidence-ranked-frontal-multiview-v4", "seed_group": group_id,
        "registration_anchors": anchors,
        "lighting_reference": lighting_reference,
        "selection_expand_groups": policy.selection_expand_groups,
        "selection_allow_quality_warnings": policy.selection_allow_quality_warnings,
        "candidate_count": len(candidates), "associated_count": len(associated),
        "selected_count": len(chosen), "view_counts": counts,
        "groups_used": dict(Counter(group_ids[face["id"]] for face in chosen)),
        "selected_face_ids": [face["id"] for face in chosen],
        "identity_decisions": decisions, "excluded": excluded,
        "identity_confirmed": False, "pose_calibrated": False,
        "baseline_face_id": max(
            (face for face in chosen if face["view"]["category"] == "front"),
            key=view_rank, default={"id": None})["id"],
        "ranking": "quality/light warnings, view suitability/detail/relative lighting; pose-fit confidence is a tiebreaker",
        "limitations": [
            "Five landmarks give approximate view bins, not dense facial detail or occlusion proof",
            "Membership remains a seed-subject hypothesis; ambiguous associations require review",
            "Front-weighted selection does not make COLMAP a semantic template fitter",
        ],
    }
    entries = []
    for face in chosen:
        relative = face["products"]["portrait"]
        width, height = (face["portrait_bounds"][i + 2] - face["portrait_bounds"][i] for i in (0, 1))
        if width <= 0 or height <= 0:
            raise ValueError("Invalid reviewed portrait dimensions")
        x, y, w, h = face["box_xywh"]
        if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError("Invalid face bounds")
        entries.append({
            "image_name": Path(relative).name,
            "face_id": face["id"], "source_id": face["source_id"],
            "source": report["sources"][face["source_id"]]["artifact"],
            "portrait": {"path": relative, "sha256": Path(relative).stem},
            "portrait_bounds": face["portrait_bounds"], "size": [width, height],
            "box_xywh": [x - face["portrait_bounds"][0], y - face["portrait_bounds"][1], w, h],
            "identity_ambiguous": face.get("identity_ambiguous", True),
            "group": group_ids[face["id"]], "view": face["view"],
            "lighting": face["lighting"],
            "landmarks_5_source": face.get("landmarks_5", []),
            "selection_weight": round(
                (1.0 if face["view"]["category"] == "front" else
                 0.35 if face["view"]["category"] == "unknown" else 0.75)
                * (0.5 + 0.5*face["view"]["confidence"])/(1 + len(face["quality"]["reasons"])), 5),
            "quality_warnings": face["quality"]["reasons"],
        })
        if cleansed is not None:
            derivative = cleansed["faces"][face["id"]]
            entries[-1].update(
                image=derivative["products"]["rgb"],
                image_name=face["id"] + ".png",
                foreground_alpha=derivative["products"]["alpha"],
                cleansing={"method": cleansed["method"], "policy": cleansed["policy"],
                           "color": derivative["color"], "quality": derivative["quality"]})
            entries[-1]["analysis"] = {
                "alignment": derivative["alignment"],
                "aligned_cutout": derivative["products"].get("aligned"),
                "native_projection_preserved": True,
            }
    mask_method = "portrait-alpha-threshold-v1" if cleansed is not None else "face-ellipse-v1"
    payload = {"schema_version": 1, "mask": mask_method,
               "curation_sha256": gallery.fingerprint, "selection": selection, "entries": entries}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    relative = Path("generation") / "inputs" / key
    root = store.resolve(str(relative))
    manifest = root / "sources.json"
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing["inputs"] != payload:
            raise ValueError("Reusable input view metadata changed")
        for artifact in existing["files"]:
            if not store.verify(artifact):
                raise ValueError("Reusable input view changed; do not silently reconstruct it")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    files = []
    try:
        (root / "images").mkdir(exist_ok=True)
        (root / "masks").mkdir(exist_ok=True)
        for entry in entries:
            image_artifact = entry.get("image", entry["portrait"])
            source = store.resolve(image_artifact["path"])
            target = root / "images" / entry["image_name"]
            if target.exists():
                if not os.path.samefile(source, target):
                    raise ValueError("Input view contains an unexpected photo")
            else:
                os.link(source, target)
            if digest(target) != image_artifact["sha256"]:
                raise ValueError("Portrait changed while creating the reusable native view")
            with Image.open(target) as image:
                if list(image.size) != entry["size"]:
                    raise ValueError("Portrait dimensions differ from extraction metadata")
            if "foreground_alpha" in entry:
                with Image.open(store.resolve(entry["foreground_alpha"]["path"])) as alpha:
                    if alpha.mode != "L" or list(alpha.size) != entry["size"]:
                        raise ValueError("Portrait alpha dimensions/mode do not match frozen inputs")
                    threshold = round(entry["cleansing"]["policy"]["native_mask_threshold"]*255)
                    mask = alpha.point(lambda value: 255 if value >= threshold else 0)
                files.append(entry["foreground_alpha"])
            else:
                mask = Image.new("L", tuple(entry["size"]), 0)
                x, y, w, h = entry["box_xywh"]
                ImageDraw.Draw(mask).ellipse((x - 0.1*w, y - 0.15*h, x + 1.1*w, y + 1.1*h), fill=255)
            mask_path = root / "masks" / (entry["image_name"] + ".png")
            mask.save(mask_path)
            files.extend([store.reference_artifact(target), store.reference_artifact(mask_path)])
        previews = {}
        review_ids = {record["face_id"] for record in decisions
                      if record["decision"].startswith("review")}
        for name, members in (("selected", chosen),
                              ("review", [face for face in candidates if face["id"] in review_ids])):
            if not members:
                continue
            tiles = []
            for face in members:
                preview_source = face["products"]["portrait"]
                if cleansed is not None:
                    derivative = cleansed["faces"].get(face["id"], {})
                    if derivative.get("status") == "ready":
                        preview_source = derivative["products"]["cutout"]["path"]
                with Image.open(store.resolve(preview_source)) as image:
                    if image.mode == "RGBA":
                        image = image.crop(image.getchannel("A").getbbox())
                    image.thumbnail((190, 195))
                    tile = image.convert("RGB")
                    if image.mode == "RGBA":
                        tile = Image.new("RGB", image.size, (85, 100, 115))
                        tile.paste(image, mask=image.getchannel("A"))
                label = f"{face['view']['category']} {group_ids[face['id']]}\n{face['id'][:12]}"
                tiles.append((label, tile))
            preview = root / f"{name}-views.jpg"
            contact_sheet(tiles).save(preview, quality=88)
            previews[name] = store.reference_artifact(preview)
            files.append(previews[name])
        result = {"key": key, "run": run, "group": group_id, "inputs": payload,
                  "selection": selection, "previews": previews,
                  "files": files, "manifest": str(manifest.relative_to(store.root)),
                  "image_storage": "reusable hardlinks to generated portraits; no original copies",
                  "identity_confirmed": False, "requires_capture_review": True}
        atomic_json(manifest, result)
        return result
    finally:
        if not manifest.is_file():
            cleanup_owned_files(store, str(relative), [])

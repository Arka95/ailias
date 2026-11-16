import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw

from ailias.generation.settings import GenerationPolicy
from ailias.imageprocessing.curation import load_filter
from ailias.imageprocessing.pipeline import component, load_run
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.resources import cleanup_owned_files
from ailias.storage import Store, atomic_json, digest


def prepare_inputs(store: Store, run: str, group_id: str, policy: GenerationPolicy) -> dict:
    report = load_run(store, run)
    group = next((g for g in report["groups"] if g["id"] == component(group_id)), None)
    if group is None:
        raise ValueError("Unknown face group")
    gallery = load_filter(store, report["backend"], ProcessingPolicy(**report["policy"]))
    if gallery is None:
        raise ValueError("Freeze manual portrait review with imageprocessing curate before generation")
    candidates = []
    for face_id in group["members"]:
        face = report["faces"][face_id]
        relative = face["products"].get("portrait")
        if relative is None or face["quality"]["reasons"]:
            continue
        portrait = store.resolve(relative)
        reviewed = gallery.hashes.get(portrait.stem)
        if reviewed is None or reviewed["label"] != "accept":
            continue
        if relative != reviewed["artifact"]["path"] or not store.verify(reviewed["artifact"]):
            raise ValueError("Reviewed portrait changed; curate again before reconstruction")
        if not store.verify(report["sources"][face["source_id"]]["artifact"]):
            raise ValueError("Original source changed since imageprocessing")
        candidates.append(face)
    candidates.sort(key=lambda f: (-f["quality"]["native_short_side"],
                                   -f["quality"]["laplacian_variance"], f["id"]))
    unique = {}
    for face in candidates:
        unique.setdefault(face["source_id"], face)
    chosen = list(unique.values())[:policy.max_images]
    if len(chosen) < policy.min_registered_images:
        raise ValueError("Too few distinct, explicitly kept, quality-eligible photos in this group")
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
        })
    payload = {"schema_version": 1, "mask": "face-ellipse-v1",
               "curation_sha256": gallery.fingerprint, "entries": entries}
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
            source = store.resolve(entry["portrait"]["path"])
            target = root / "images" / entry["image_name"]
            if target.exists():
                if not os.path.samefile(source, target):
                    raise ValueError("Input view contains an unexpected photo")
            else:
                os.link(source, target)
            if digest(target) != entry["portrait"]["sha256"]:
                raise ValueError("Portrait changed while creating the reusable native view")
            with Image.open(target) as image:
                if list(image.size) != entry["size"]:
                    raise ValueError("Portrait dimensions differ from extraction metadata")
            mask = Image.new("L", tuple(entry["size"]), 0)
            x, y, w, h = entry["box_xywh"]
            ImageDraw.Draw(mask).ellipse((x - 0.1*w, y - 0.15*h, x + 1.1*w, y + 1.1*h), fill=255)
            mask_path = root / "masks" / (entry["image_name"] + ".png")
            mask.save(mask_path)
            files.extend([store.reference_artifact(target), store.reference_artifact(mask_path)])
        result = {"key": key, "run": run, "group": group_id, "inputs": payload,
                  "files": files, "manifest": str(manifest.relative_to(store.root)),
                  "image_storage": "reusable hardlinks to generated portraits; no original copies",
                  "identity_confirmed": False, "requires_capture_review": True}
        atomic_json(manifest, result)
        return result
    finally:
        if not manifest.is_file():
            cleanup_owned_files(store, str(relative), [])

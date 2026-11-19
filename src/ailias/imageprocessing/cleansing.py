"""Versioned appearance derivatives; reviewed portraits are never modified."""

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PIL import Image

from ailias.imageprocessing.assets import store_png
from ailias.imageprocessing.curation import load_filter
from ailias.imageprocessing.features import contact_sheet
from ailias.common.identifiers import validate_identifier as component, generate_run_id as new_id
from ailias.imageprocessing.pipeline import load_run
from ailias.imageprocessing.image_io import read_oriented_image as read_image
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.imageprocessing.observations import align_head, assess_lighting, face_lighting
from ailias.imageprocessing.subjects import associate_subject, view_rank
from ailias.imageprocessing.views import estimate_view
from ailias.resource_management import finalize_job_resources
from ailias.settings import PROJECT_ROOT
from ailias.storage import atomic_json, digest

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "cleansing.json"
METHOD = "scene-neutrals-subject-exposure-aligned-head-v4"


@dataclass(frozen=True)
class CleansingPolicy:
    neutral_saturation_max: float = 0.2
    neutral_min_fraction: float = 0.005
    white_balance_gain_limit: float = 1.18
    exposure_ev_limit: float = 0.35
    alpha_background_cutoff: float = 0.02
    alpha_foreground_cutoff: float = 0.98
    native_mask_threshold: float = 0.5
    min_face_coverage: float = 0.9

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{field.name} must be finite and positive")
        if (not self.alpha_background_cutoff < self.native_mask_threshold
                < self.alpha_foreground_cutoff < 1
                or not 1 <= self.white_balance_gain_limit <= 1.5
                or self.exposure_ev_limit > 1
                or max(self.neutral_saturation_max, self.neutral_min_fraction,
                       self.min_face_coverage) > 1):
            raise ValueError("Unsafe cleansing policy bounds")


def load_policy(path=DEFAULT_CONFIG):
    return CleansingPolicy(**json.loads(path.read_text(encoding="utf-8")))


def linear_rgb(values):
    import numpy as np
    rgb = np.asarray(values, dtype=np.float32)/255
    return np.where(rgb <= 0.04045, rgb/12.92, ((rgb + 0.055)/1.055)**2.4)


def encode_rgb(values):
    import numpy as np
    rgb = np.clip(values, 0, 1)
    encoded = np.where(rgb <= 0.0031308, rgb*12.92, 1.055*rgb**(1/2.4) - 0.055)
    return Image.fromarray(np.rint(encoded*255).astype(np.uint8))


def white_balance(scene, boxes, policy):
    import numpy as np

    sample = scene.copy()
    sample.thumbnail((512, 512))
    values = np.asarray(sample, dtype=np.float32)/255
    maximum, minimum = values.max(2), values.min(2)
    neutral = ((maximum > 0.2) & (maximum < 0.9)
               & ((maximum - minimum)/np.maximum(maximum, 1e-6) <= policy.neutral_saturation_max))
    sx, sy = sample.width/scene.width, sample.height/scene.height
    for x, y, width, height in boxes:
        x0, y0 = max(0, int(x*sx)), max(0, int(y*sy))
        x1, y1 = min(sample.width, math.ceil((x + width)*sx)), min(sample.height, math.ceil((y + height)*sy))
        neutral[y0:y1, x0:x1] = False
    count = int(neutral.sum())
    record = {"method": "low-saturation scene neutrals excluding detected faces",
              "calibrated": False, "neutral_pixels": count, "gains": [1., 1., 1.]}
    if count < max(64, neutral.size*policy.neutral_min_fraction):
        return {**record, "status": "skipped_insufficient_scene_neutrals"}
    pixels = linear_rgb(np.asarray(sample))[neutral]
    chromaticity = pixels/np.maximum(pixels.mean(1, keepdims=True), 1e-6)
    center = np.median(chromaticity, axis=0)
    spread = float(np.quantile(np.abs(chromaticity - center), 0.9))
    gains = 1/center
    gains /= np.exp(np.log(gains).mean())
    if spread > 0.12 or max(gains.max(), 1/gains.min()) > policy.white_balance_gain_limit:
        return {**record, "status": "skipped_ambiguous_or_excessive_cast", "chromaticity_spread": spread}
    return {**record, "status": "estimated_scene_neutrals", "gains": gains.tolist(),
            "chromaticity_spread": spread}


def core_slice(bounds, size):
    x, y, width, height = bounds
    left = max(0, min(size[0] - 1, int(x + width*0.25)))
    right = max(left + 1, min(size[0], math.ceil(x + width*0.75)))
    top = max(0, min(size[1] - 1, int(y + height*0.2)))
    bottom = max(top + 1, min(size[1], math.ceil(y + height*0.8)))
    return slice(top, bottom), slice(left, right)


def appearance(image, bounds, balance, policy, target=None):
    import numpy as np

    values = linear_rgb(np.asarray(image))*np.asarray(balance["gains"], dtype=np.float32)
    core = values[core_slice(bounds, image.size)]
    luminance = float(np.median(core @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
    ev = 0.
    status = "skipped_no_subject_reference"
    if target is not None and luminance > 1e-5:
        ev = float(np.clip(math.log2(target/luminance),
                           -policy.exposure_ev_limit, policy.exposure_ev_limit))
        status = "relative_subject_harmonization"
    before = float((np.max(values, axis=2) >= 1).mean())
    corrected = values*2**ev
    if float((corrected.max(2) >= 1).mean()) - before > 0.01:
        corrected, ev, status = values, 0., "skipped_highlight_clipping"
    return encode_rgb(corrected), {
        "white_balance": balance, "exposure_status": status, "exposure_ev": ev,
        "reference_luminance": target, "core_luminance": luminance,
        "intrinsic_albedo": False, "skin_tone_calibrated": False,
    }


def foreground(image, alpha, bounds, policy):
    import numpy as np

    if alpha.mode != "L" or alpha.size != image.size:
        raise ValueError("Matting must return native-size L alpha, without changing crop geometry")
    values = np.asarray(alpha).copy()
    values[values <= round(policy.alpha_background_cutoff*255)] = 0
    values[values >= round(policy.alpha_foreground_cutoff*255)] = 255
    mask = values >= round(policy.native_mask_threshold*255)
    coverage = float(mask[core_slice(bounds, image.size)].mean())
    fraction = float(mask.mean())
    details = {"face_core_coverage": coverage, "foreground_fraction": fraction,
               "fractional_alpha_fraction": float(((values > 0) & (values < 255)).mean()),
               "native_mask_threshold": policy.native_mask_threshold,
               "quality_calibrated": False}
    if coverage < policy.min_face_coverage or not 0.03 < fraction < 0.99:
        return None, {**details, "status": "review", "reason": "unreliable_foreground_or_face_coverage"}
    native = np.asarray(image).copy()
    rgba = np.dstack((native.copy(), values))
    rgba[values == 0, :3] = 0
    native[~mask] = 0
    return (Image.fromarray(native), Image.fromarray(rgba), Image.fromarray(values)), {**details, "status": "ready"}


def load_cleansing(store, run, curation_sha256):
    pointer = store.resolve(str(Path("imageprocessing") / "cleansing" / f"{component(run)}.json"))
    if not pointer.is_file():
        raise ValueError("Portrait cleansing is required; run imageprocessing cleanse first")
    artifact = json.loads(pointer.read_text(encoding="utf-8"))
    if not store.verify(artifact):
        raise ValueError("Cleansing manifest changed; run imageprocessing cleanse again")
    result = json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    if (result.get("status") != "complete" or result.get("run") != run
            or result.get("method") != METHOD or result.get("curation_sha256") != curation_sha256):
        raise ValueError("Cleansing is stale for current review; run imageprocessing cleanse again")
    return result


def cleanse_run(store, run, policy, backend, group=None, progress=None):
    import numpy as np

    report = load_run(store, run)
    processing = ProcessingPolicy(**report["policy"])
    gallery = load_filter(store, report["backend"], processing)
    candidates, excluded = [], []
    for face in report["faces"].values():
        relative = face["products"].get("portrait")
        if not relative:
            continue
        path = store.resolve(relative)
        if gallery and (path.stem not in gallery.hashes or gallery.hashes[path.stem]["label"] != "accept"):
            continue
        if "portrait_contains_other_face" in face["quality"]["reasons"]:
            excluded.append({"face_id": face["id"], "reason": "portrait_contains_other_face"})
            continue
        if digest(path) != path.stem:
            raise ValueError("Portrait changed before cleansing")
        candidates.append({**face, "view": estimate_view(
            face, report["sources"][face["source_id"]].get("metadata", {}))})
    if group:
        group = component(group)
        if not any(item["id"] == group for item in report["groups"]):
            raise ValueError("Unknown face group")
        candidates, decisions = associate_subject(candidates, report["groups"], group, processing)
        excluded.extend(item for item in decisions if item["decision"] not in {"seed", "associated"})
    if not candidates:
        raise ValueError("No current usable portraits to cleanse")
    job = new_id()
    relative = Path("imageprocessing") / "cleansing" / job
    directory = store.resolve(str(relative))
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory / "manifest.json"
    result = {
        "schema_version": 1, "method": METHOD, "job": job, "run": run, "group": group,
        "status": "running", "curation_sha256": gallery.fingerprint if gallery else None,
        "policy": asdict(policy), "backend": backend.provenance, "faces": {}, "excluded": excluded,
        "source_run": store.reference_artifact(store.resolve(
            str(Path("imageprocessing") / "runs" / run / "manifest.json"))),
        "requires_visual_review": True, "intrinsic_albedo": False,
    }
    created, scene_balances, prepared = [], {}, []
    group_ids = {key: item["id"] for item in report["groups"] for key in item["members"]}
    atomic_json(manifest, result)
    try:
        for face in candidates:
            source_id = face["source_id"]
            source = report["sources"][source_id]["artifact"]
            if source_id not in scene_balances:
                if not store.verify(source):
                    raise ValueError("Original source changed before cleansing")
                scene, _ = read_image(store.artifact_path(source))
                boxes = [other["box_xywh"] for other in report["faces"].values()
                         if other["source_id"] == source_id]
                scene_balances[source_id] = white_balance(scene, boxes, policy)
                if not store.verify(source):
                    raise ValueError("Original source changed during cleansing")
                del scene
            x, y, width, height = face["box_xywh"]
            bounds = [x - face["portrait_bounds"][0], y - face["portrait_bounds"][1], width, height]
            with Image.open(store.resolve(face["products"]["portrait"])) as portrait:
                rows, columns = core_slice(bounds, portrait.size)
                core = portrait.crop((columns.start, rows.start, columns.stop, rows.stop)).convert("RGB")
                values = linear_rgb(np.asarray(core))*np.asarray(
                    scene_balances[source_id]["gains"], dtype=np.float32)
                luminance = float(np.median(values @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
            prepared.append((face, bounds, luminance, group or group_ids[face["id"]]))
        references = {}
        for scope in {item[3] for item in prepared}:
            observations = [item[2] for item in prepared if item[3] == scope
                            and item[0]["view"]["category"] == "front"
                            and not item[0]["quality"]["reasons"] and item[2] > 1e-5]
            if len(observations) >= 3:
                references[scope] = float(np.median(observations))
        result["exposure_references"] = {
            "method": "median linear face-core luminance of at least three clean frontal subject hypotheses",
            "targets": references, "identity_confirmed": False,
        }
        for index, (face, bounds, _, scope) in enumerate(prepared):
            path = store.resolve(face["products"]["portrait"])
            if digest(path) != path.stem:
                raise ValueError("Portrait changed during cleansing")
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            alpha = backend.matte(image)
            from ailias.imageprocessing.faceparsing import head_alpha
            parts = backend.parse(image, bounds)
            alpha, semantic = head_alpha(alpha, parts, bounds)
            corrected, color = appearance(image, bounds, scene_balances[face["source_id"]],
                                           policy, references.get(scope))
            products, quality = foreground(corrected, alpha, bounds, policy)
            if semantic["review_reason"]:
                products = None
                quality = {**quality, "status": "review", "reason": semantic["review_reason"]}
            record = {
                "portrait": store.reference_artifact(path), "source": report["sources"][face["source_id"]]["artifact"],
                "size": list(image.size), "portrait_bounds": face["portrait_bounds"],
                "status": quality["status"], "quality": quality, "color": color,
                "semantic_region": semantic, "products": {}, "view": face["view"],
                "subject_scope": scope, "quality_warnings": face["quality"]["reasons"],
                "source_name": store.artifact_path(report["sources"][face["source_id"]]["artifact"]).name,
                "lighting": face_lighting(corrected, parts, alpha, bounds),
                "alignment": {"status": "not_published"},
            }
            if products:
                for name, value, category in zip(("rgb", "cutout", "alpha"), products,
                                                  ("cleansed", "cutouts", "alphas")):
                    product = store_png(store, value, category, directory / "work", created)
                    record["products"][name] = store.reference_artifact(store.resolve(product))
                landmarks = [[point[0] - face["portrait_bounds"][0], point[1] - face["portrait_bounds"][1]]
                             for point in face.get("landmarks_5", [])]
                aligned, record["alignment"] = align_head(products[1], landmarks)
                if aligned is not None:
                    source_to_portrait = np.eye(3)
                    source_to_portrait[:2, 2] = -np.asarray(face["portrait_bounds"][:2])
                    source_to_aligned = np.asarray(record["alignment"]["portrait_to_aligned"]) @ source_to_portrait
                    record["alignment"]["source_to_aligned"] = source_to_aligned.tolist()
                    record["alignment"]["aligned_to_source"] = np.linalg.inv(source_to_aligned).tolist()
                    product = store_png(store, aligned, "aligned", directory / "work", created)
                    record["products"]["aligned"] = store.reference_artifact(store.resolve(product))
            result["faces"][face["id"]] = record
            atomic_json(manifest, result)
            if progress:
                progress({"processed": index + 1, "total": len(prepared), "stage": "cleansing"})
        for record in result["faces"].values():
            if not store.verify(record["source"]) or not store.verify(record["portrait"]):
                raise ValueError("Source or reviewed portrait changed during cleansing")
        result["lighting_references"] = assess_lighting(result["faces"])
        ordered = sorted(
            [dict(face, lighting=result["faces"][face["id"]]["lighting"]) for face in candidates
             if result["faces"][face["id"]]["status"] == "ready"],
            key=lambda face: (face["view"]["category"] == "front", view_rank(face)), reverse=True)
        result["preview_pages"] = []
        for offset in range(0, len(ordered), 36):
            page = ordered[offset:offset + 36]
            tiles = []
            for face in page:
                record = result["faces"][face["id"]]
                artifact = record["products"].get("aligned", record["products"]["cutout"])
                with Image.open(store.artifact_path(artifact)) as cutout:
                    cutout.thumbnail((190, 195))
                    tile = Image.new("RGB", cutout.size, (85, 100, 115))
                    tile.paste(cutout, mask=cutout.getchannel("A"))
                name = record["source_name"]
                tiles.append((name[:28] + "\n" + name[28:52], tile))
            preview = directory / ("cleansed-preview.jpg" if offset == 0 else f"cleansed-preview-{offset//36 + 1}.jpg")
            contact_sheet(tiles).save(preview, quality=88)
            result["preview_pages"].append({"artifact": store.reference_artifact(preview),
                                             "face_ids": [face["id"] for face in page]})
        if result["preview_pages"]:
            result["preview"] = result["preview_pages"][0]["artifact"]
        result["summary"] = dict(Counter(record["status"] for record in result["faces"].values()))
        result["summary"]["alignment"] = dict(Counter(
            record["alignment"]["status"] for record in result["faces"].values()))
        result["summary"]["lighting_warnings"] = dict(Counter(
            warning for record in result["faces"].values() for warning in record["lighting"]["warnings"]))
        result["summary"]["white_balance"] = dict(Counter(
            record["color"]["white_balance"]["status"] for record in result["faces"].values()))
        result["summary"]["exposure"] = dict(Counter(
            record["color"]["exposure_status"] for record in result["faces"].values()))
        result["status"] = "complete"
    finally:
        if result["status"] == "running":
            result["status"] = "failed"
        retained = [str(manifest.relative_to(store.root))]
        for record in result["faces"].values():
            retained.extend(artifact["path"] for artifact in record["products"].values())
        if result.get("preview"):
            retained.append(result["preview"]["path"])
        retained.extend(page["artifact"]["path"] for page in result.get("preview_pages", []))
        finalize_job_resources(
            store, result, manifest, [store.resolve(path) for path in retained],
            retain_metadata=False, created_assets=created,
            automatic_retention=False,
        )
    if result["status"] == "complete":
        atomic_json(store.resolve(str(Path("imageprocessing") / "cleansing" / f"{run}.json")),
                    store.reference_artifact(manifest))
    return {**result, "manifest": str(manifest)}

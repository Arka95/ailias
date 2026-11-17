"""Build auditable photo appearance for approved geometry without rerunning shape fitting."""

import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from ailias.facefitting.model import HeadModel
from ailias.facefitting.pipeline import inspect_fit
from ailias.facefitting.texture import choose_texture_views, uv_surface_map, validate_material_textures
from ailias.imageprocessing.pipeline import component
from ailias.resources import finalize_resources
from ailias.settings import PROJECT_ROOT
from ailias.storage import atomic_json

DEFAULT_APPEARANCE_CONFIG = PROJECT_ROOT / "config" / "appearance.json"


def load_appearance_policy(path=DEFAULT_APPEARANCE_CONFIG):
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not re.fullmatch(r"cuda:[0-9]+", policy.get("device", "")):
        raise ValueError("Photo baking requires an explicitly selected CUDA device")
    for key, low, high in (
        ("texture_size", 256, 4096), ("photo_size", 256, 2048), ("max_views", 1, 64),
        ("render_samples", 1, 128), ("hair_strands", 0, 50000),
        ("hair_segments", 3, 16), ("seed", 0, 2**31-1),
    ):
        if type(policy.get(key)) is not int or not low <= policy[key] <= high:
            raise ValueError(f"{key} must be an integer in {low}..{high}")
    return policy


def approved_fit(store, job):
    report = inspect_fit(store, job)
    if report["integrity_problems"]:
        raise ValueError("Approved fit dependencies changed; do not texture stale geometry")
    approval_path = store.resolve(str(Path("facefitting") / "reviews" /
                                      f"{component(job)}-clay-approval.json"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (approval.get("decision") != "approved" or approval.get("scope") != "neutral_clay_geometry"
            or approval.get("mesh") != report["artifacts"]["mesh"]):
        raise ValueError("Explicit approval of this exact neutral geometry is required")
    return report, store.reference_artifact(approval_path)


def inspect_appearance(store, revision):
    path = store.resolve(str(Path("facefitting") / "appearance" / component(revision) / "manifest.json"))
    report = json.loads(path.read_text(encoding="utf-8"))
    dependencies = [report["approval"], report["approved_mesh"], report["fit_manifest"],
                    *report.get("input_dependencies", []), *report["artifacts"].values()]
    return {**report, "integrity_problems": [item["path"] for item in dependencies if not store.verify(item)]}


def run_appearance(store, job, policy, generation, tools):
    started = time.monotonic()
    fitted, approval = approved_fit(store, job)
    revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = store.resolve(str(Path("facefitting") / "appearance" / revision))
    root.mkdir(parents=True, exist_ok=False)
    path = root / "manifest.json"
    report = {
        "schema_version": 1, "revision": revision, "fit_job": job, "policy": policy,
        "status": "running", "current_step": "select_photo_evidence", "artifacts": {},
        "approval": approval, "approved_mesh": fitted["artifacts"]["mesh"],
        "fit_manifest": store.reference_artifact(store.resolve(str(
            Path("facefitting") / "runs" / component(job) / "manifest.json"))),
        "limitations": [
            "Photo-derived colour retains residual photographed lighting; not intrinsic albedo",
            "Unobserved texture is neutral-filled and explicitly absent from the coverage map",
            "Hair colour/roots/length use photo evidence; strand arrangement and hidden hair are approximations",
            "Generic eye/teeth geometry remains from the approved clay model",
            "No MetaHuman conform, rig, animation or final appearance approval is implied",
        ],
    }
    atomic_json(path, report)
    temporary_photos = []
    try:
        inputs = json.loads(store.artifact_path(fitted["inputs"]).read_text(encoding="utf-8"))
        entries = {entry["face_id"]: entry for entry in inputs["inputs"]["entries"]}
        observations = json.loads(store.artifact_path(fitted["artifacts"]["observations"]).read_text())
        observations = {item["face_id"]: item for item in observations["faces"]}
        cameras = json.loads(store.artifact_path(fitted["artifacts"]["cameras"]).read_text())
        selected = choose_texture_views(cameras["views"], entries, observations, policy["max_views"])
        report["selected_views"] = [view["face_id"] for view in selected]
        dependencies = [fitted["inputs"], *[asset["artifact"] for asset in fitted["assets"].values()]]
        for view in selected:
            dependencies += [entries[view["face_id"]][key] for key in ("image", "foreground_alpha", "source")]
        if not all(store.verify(item) for item in dependencies):
            raise ValueError("Appearance source evidence changed")
        report["input_dependencies"] = dependencies
        for name in ("appearance.py", "texture.py", "texture_cuda.py"):
            report["artifacts"][name] = store.copy_artifact(Path(__file__).with_name(name),
                                                            str(root.relative_to(store.root) / "worker"))
        worker = store.copy_artifact(Path(__file__).with_name("semantics_worker.py"),
                                     str(root.relative_to(store.root) / "worker"))
        report["artifacts"]["semantics_worker"] = worker
        request = root / "semantics-request.json"
        atomic_json(request, {
            "data_root": str(store.root.parent), "alias": store.root.name,
            "device": policy["device"], "photo_size": policy["photo_size"], "output": str(root),
            "photos": [{"entry": entries[view["face_id"]]} for view in selected],
        })
        report["current_step"] = "fresh_cuda_photo_semantics"
        atomic_json(path, report)
        temporary_photos = [root / f"photo-{index:02}.npz" for index in range(len(selected))]
        tools.paths["python"] = Path(sys.executable)
        report["semantics_runtime"] = tools.run(
            "python", "appearance-semantics", [store.artifact_path(worker), request], root)
        semantics = json.loads((root / "semantics.json").read_text())
        dependencies += [semantics["parser"]["model"], semantics["parser"]["license_artifact"]]
        model = HeadModel(store.artifact_path(fitted["assets"]["gnm"]["artifact"]),
                          store.artifact_path(fitted["assets"]["correspondence"]["artifact"]),
                          fitted["policy"]["expression_modes"])
        with np.load(store.artifact_path(fitted["assets"]["gnm"]["artifact"]), allow_pickle=False) as data:
            uvs = data["triangle_uvs"].copy()
        with np.load(store.artifact_path(fitted["artifacts"]["parameters"]), allow_pickle=False) as data:
            identity = data["identity"].copy()
        neutral = model.neutral(identity)
        points = model.sample(neutral)
        supported_skin = (model.groups["skin_exterior"] > .5) & (neutral[:, 1] >= points[152, 1]-.005)
        exterior = np.all(supported_skin[model.triangles], axis=1)
        report["current_step"] = "cuda_uv_projection_and_visibility"
        atomic_json(path, report)
        mapping = uv_surface_map(uvs, policy["texture_size"], exterior)
        from ailias.facefitting.texture_cuda import bake
        baked = bake(model, identity, mapping, selected, entries, semantics["photos"], policy["device"])
        rows, columns, _, _ = mapping
        size = policy["texture_size"]
        texture = np.broadcast_to(baked["skin_color"], (size, size, 3)).copy()
        coverage = np.zeros((size, size), dtype=bool)
        texture[rows, columns] = np.where(baked["coverage"][:, None],
                                          baked["color"], baked["skin_color"])
        coverage[rows, columns] = baked["coverage"]
        if not coverage.any():
            raise ValueError("No visible photo texels survived semantic/visibility checks")
        distance, nearest = distance_transform_edt(~coverage, return_indices=True)
        padding = (~coverage) & (distance <= 8)
        texture[padding] = texture[nearest[0][padding], nearest[1][padding]]
        Image.fromarray(np.uint8(np.clip(texture, 0, 1)*255)).save(root / "skin-basecolor.png")
        Image.fromarray(np.uint8(coverage)*255).save(root / "texture-coverage.png")
        np.savez_compressed(root / "appearance-geometry.npz",
                            vertices=neutral, triangles=model.triangles, triangle_uvs=uvs,
                            skin_exterior=model.groups["skin_exterior"],
                            eye_exteriors=model.groups["eye_exteriors"],
                            hair_probability=baked["hair_probability"], hair_observed=baked["hair_observed"],
                            eye_level=points[[33, 263], 1].mean())
        atomic_json(root / "appearance-evidence.json", {
            "texture": baked["report"], "parser": semantics["parser"],
            "hair": {"color_srgb": baked["hair_color"].tolist(), "length": baked["hair_length"],
                     "length_units": "uncalibrated GNM model units",
                     "strand_layout": "procedural approximation, not reconstructed individual strands"},
        })
        render_worker = store.copy_artifact(Path(__file__).with_name("appearance_worker.py"),
                                            str(root.relative_to(store.root) / "worker"))
        request = root / "appearance-request.json"
        atomic_json(request, {
            "output": str(root), "geometry": str(root / "appearance-geometry.npz"),
            "texture": str(root / "skin-basecolor.png"), "policy": policy,
            "device": generation.blender_device, "size": generation.render_size,
            "hair_color": baked["hair_color"].tolist(), "hair_length": baked["hair_length"],
        })
        report["current_step"] = "blender_photo_material_and_hair_review"
        atomic_json(path, report)
        report["render_runtime"] = tools.run("blender", "appearance-review", [
            "--background", str(store.artifact_path(fitted["artifacts"]["scene"])),
            "--disable-autoexec", "--python-exit-code", "1",
            "--python", str(store.artifact_path(render_worker)), "--", str(request),
        ], root)
        for material in ("textured-head.mtl", "head-for-unreal.mtl"):
            validate_material_textures(root / material, root / "skin-basecolor.png")
        with (root / "head-for-unreal.obj").open() as exported:
            exported_vertices = np.asarray([np.fromstring(line[2:], sep=" ")
                                             for line in exported if line.startswith("v ")])
        if (exported_vertices.shape != neutral.shape
                or not np.allclose(exported_vertices, neutral[:, [2, 0, 1]]*100, atol=1e-4, rtol=0)):
            raise ValueError("Unreal export changed the approved head beyond declared axes/uniform scale")
        products = {
            "texture": "skin-basecolor.png", "coverage": "texture-coverage.png",
            "geometry": "appearance-geometry.npz", "evidence": "appearance-evidence.json",
            "scene": "textured-head.blend", "mesh": "textured-head.obj", "materials": "textured-head.mtl",
            "head_only_mesh": "head-for-unreal.obj", "head_only_materials": "head-for-unreal.mtl",
            "blender_runtime": "appearance-runtime.json",
            "render_request": "appearance-request.json",
        }
        previews = []
        for angle in (-45, 0, 45, 90):
            name = f"appearance-{angle}.png"
            with Image.open(root / name) as image:
                image.verify()
            products[name] = name
            previews.append(root / name)
        side = generation.render_size
        sheet = Image.new("RGB", (side*4, side+30), "#181818")
        draw = ImageDraw.Draw(sheet)
        for index, (angle, preview) in enumerate(zip((-45, 0, 45, 90), previews)):
            with Image.open(preview) as image:
                sheet.paste(image.convert("RGB"), (index*side, 30))
            draw.text((index*side+8, 8), f"PHOTO COLOUR + APPROXIMATE HAIR / {angle}", fill="white")
        sheet.save(root / "appearance-comparison.jpg", quality=94)
        products["comparison"] = "appearance-comparison.jpg"
        report["artifacts"].update({key: store.reference_artifact(root / value)
                                    for key, value in products.items()})
        report["artifacts"]["render_worker"] = render_worker
        if (root / "hair-groom.abc").is_file():
            report["artifacts"]["groom"] = store.reference_artifact(root / "hair-groom.abc")
        report["texture"] = baked["report"]
        if not all(store.verify(item) for item in dependencies+[approval, report["approved_mesh"]]):
            raise ValueError("Approval or source evidence changed during appearance generation")
        report.update(status="awaiting_appearance_review", current_step="review_photo_texture_and_hair")
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        for temporary in temporary_photos:
            if temporary.parent.resolve() == root.resolve() and temporary.name.startswith("photo-"):
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        semantics_path = root / "semantics.json"
        if semantics_path.is_file():
            try:
                saved = json.loads(semantics_path.read_text())
                for photo in saved.get("photos", []):
                    photo.pop("data", None)
                saved["temporary_photo_arrays_removed"] = not any(p.exists() for p in temporary_photos)
                atomic_json(semantics_path, saved)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                report.setdefault("error", {"type": type(error).__name__,
                                            "message": "Invalid semantics receipt"})
                report["status"] = "failed"
            if semantics_path.stat().st_size:
                report["artifacts"]["semantics"] = store.reference_artifact(semantics_path)
        report["total_seconds"] = round(time.monotonic()-started, 3)
        finalize_resources(store, report, path)
        report["temporary_photo_arrays_removed"] = not any(p.exists() for p in temporary_photos)
        atomic_json(path, report)
    return report

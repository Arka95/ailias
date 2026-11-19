"""Build a fresh textured Blender head and export the geometry consumed by Unreal."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from ailias.head_reconstruction.appearance.photo_texture_baking import project_photo_evidence, write_texture_atlas
from ailias.head_reconstruction.appearance.uv_rasterization import uv_surface_map, validate_material_textures
from ailias.head_reconstruction.blender.clay_rendering import render_clay_portrait
from ailias.storage import atomic_json


def create_blender_head(store, root, model, fitted, views, evidence, uvs, tools, generation, policy):
    root.mkdir(exist_ok=False)
    report = {"status": "running", "artifacts": {}, "previous_scene_reused": False}
    try:
        vertices = model.neutral(fitted["identity"])
        points = model.sample(vertices)
        supported = ((model.groups["skin_exterior"] > .5)
                     & (vertices[:, 1] >= points[152, 1]-.005))
        exterior = np.all(supported[model.triangles], axis=1)
        mapping = uv_surface_map(uvs, policy["texture_size"], exterior)
        baked, selected = project_photo_evidence(
            store, root, model, fitted["identity"], mapping, fitted, views, evidence, policy["device"],
            maximum_views=policy["max_views"], photo_size=policy["photo_size"],
        )
        texture, coverage = write_texture_atlas(
            root, mapping, baked, policy["texture_size"],
            texture_name="skin-basecolor.png", coverage_name="texture-coverage.png",
        )
        geometry = root / "appearance-geometry.npz"
        np.savez_compressed(
            geometry, vertices=vertices, triangles=model.triangles, triangle_uvs=uvs,
            skin_exterior=model.groups["skin_exterior"], eye_exteriors=model.groups["eye_exteriors"],
            hair_probability=baked["hair_probability"], hair_observed=baked["hair_observed"],
            eye_level=points[[33, 263], 1].mean(),
        )
        report["texture"] = baked["report"]
        report["selected_views"] = [view["face_id"] for view in selected]
        clay, report["clay_runtime"] = render_clay_portrait(
            store, root, model, vertices, tools, generation)
        report["artifacts"].update({"clay_" + key: value for key, value in clay.items()})
        worker = store.copy_artifact(
            Path(__file__).with_name("appearance_scene_worker.py"),
            str(root.relative_to(store.root) / "worker"),
        )
        report["artifacts"]["appearance_worker"] = worker
        request = root / "appearance-request.json"
        atomic_json(request, {
            "output": str(root), "geometry": str(geometry), "texture": str(texture), "policy": policy,
            "device": generation.blender_device, "size": generation.render_size,
            "hair_color": baked["hair_color"].tolist(), "hair_length": baked["hair_length"],
        })
        report["appearance_runtime"] = tools.run("blender", "appearance-review", [
            "--background", str(store.artifact_path(clay["scene"])), "--disable-autoexec",
            "--python-exit-code", "1", "--python", str(store.artifact_path(worker)), "--", str(request),
        ], root)
        for material in ("textured-head.mtl", "head-for-unreal.mtl"):
            validate_material_textures(root / material, texture)
        mesh = root / "head-for-unreal.obj"
        with mesh.open(encoding="utf-8") as exported:
            actual = np.asarray([np.fromstring(line[2:], sep=" ")
                                 for line in exported if line.startswith("v ")])
        expected = vertices[:, [2, 0, 1]]*100
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-4, rtol=0):
            raise RuntimeError("Blender export changed fitted head geometry or its declared coordinate basis")
        runtime = json.loads((root / "appearance-runtime.json").read_text(encoding="utf-8"))
        if (runtime["geometry_maximum_deviation"] > 1e-6
                or runtime["cycles_device"] != "GPU"):
            raise RuntimeError("Blender did not preserve the new fitted geometry on the requested GPU")
        if policy["hair_strands"] and runtime["hair"].get("strands", 0) <= 0:
            raise RuntimeError("Blender did not create the requested photo-supported hair")
        if not policy["hair_strands"] and (
            runtime["hair"].get("status") != "disabled"
            or runtime["hair"].get("strands", 0) != 0
            or runtime.get("hair_objects")
            or runtime.get("groom", "not_exported") != "not_exported"
        ):
            raise RuntimeError("Blender created hair despite the disabled hair policy")
        for angle in (-45, 0, 45, 90):
            with Image.open(root / f"appearance-{angle}.png") as image:
                image.verify()
        scene = root / "textured-head.blend"
        if not scene.is_file() or not scene.stat().st_size:
            raise RuntimeError("Blender did not save its new textured scene")
        report["artifacts"].update(
            scene=store.reference_artifact(scene), head_only_mesh=store.reference_artifact(mesh),
            texture=store.reference_artifact(texture), coverage=store.reference_artifact(coverage),
        )
        report.update(status="blender_head_created", geometry_preserved=True, hair=runtime["hair"],
                      groom_transfer="UE uses evidence-mapped native grooms, not the approximate Alembic strands")
        return report
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        primary = sys.exception()
        for path in root.iterdir():
            try:
                if path.is_file() and path.name != "manifest.json":
                    report["artifacts"][path.name] = store.reference_artifact(path)
            except OSError as error:
                if primary is None:
                    raise
                primary.add_note("Blender artifact retention also failed: " + type(error).__name__)
        try:
            atomic_json(root / "manifest.json", report)
        except OSError as error:
            if primary is None:
                raise
            primary.add_note("Blender manifest persistence also failed: " + type(error).__name__)

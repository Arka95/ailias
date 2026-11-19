"""Orchestrate camera-matched local Character drafts while preserving approved source heads."""

import json
import uuid
from pathlib import Path
from ailias.common.code_assets import module_source

import numpy as np
from PIL import Image

from ailias.head_reconstruction.artifacts import inspect_head_appearance
from ailias.generation.unreal import asset_path, readiness
from ailias.common.identifiers import validate_identifier as component
from ailias.resource_management import finalize_job_resources
from ailias.storage import atomic_json


def conform_head(store, import_job, editor, project, generation, tools, *, baseline=None,
                 reference_face_id=None, source_character=None):
    imported_path = store.resolve(str(Path("facefitting") / "unreal" /
                                     component(import_job) / "manifest.json"))
    imported = json.loads(imported_path.read_text())
    if imported["status"] != "static_head_imported_for_review":
        raise ValueError("Head-only conform requires a verified static-head import")
    dependencies = [imported["source"], *imported["artifacts"].values()]
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Imported head dependencies changed")
    source = inspect_head_appearance(store, imported["revision"])
    if source["integrity_problems"]:
        raise ValueError("Appearance source changed")
    tracking_scene = source["artifacts"]["scene"]
    baseline_path = baseline or store.resolve("facefitting\\working-baseline.json")
    selected = None
    if baseline is not None or baseline_path.is_file():
        selected = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
        if (selected.get("status") != "owner_selected_working_baseline"
                or selected.get("appearance_revision") != imported["revision"]):
            raise ValueError("Selected Blender baseline must belong to this imported appearance")
        references = selected.get("artifacts", {})
        if references.get("source_scene", {}).get("sha256") != tracking_scene["sha256"]:
            raise ValueError("Blender baseline does not derive from the imported appearance scene")
        baseline_reference = store.reference_artifact(baseline_path)
        dependencies.extend([baseline_reference, *references.values()])
        if not all(store.verify(item) for item in dependencies):
            raise ValueError("Selected Blender baseline dependencies changed")
        tracking_scene = references["scene"]
    probe = readiness(editor, project)
    if not project or project.resolve() != Path(imported["readiness"]["project"]).resolve():
        raise ValueError("Use the project containing the verified head import")
    if source_character:
        from ailias.metahuman.character_features import character_package
        source_character = asset_path(source_character)
        dependencies.append(store.reference_artifact(character_package(project, source_character)))
    revision = uuid.uuid4().hex
    root = store.resolve(str(Path("facefitting") / "conforms" / revision))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    report = {
        "revision": revision, "import_job": import_job, "status": "blocked" if probe["blockers"] else "running",
        "readiness": probe, "artifacts": {}, "input_dependencies": dependencies,
        "source_import": store.reference_artifact(imported_path),
        "cloud_requested": False, "rig_requested": False, "likeness_verified": False,
        "tracking_source": "original_photo" if reference_face_id else "blender_preview",
        "source_character": source_character,
        "limitations": [
            "Local head-only Character draft, not an assembled or validated MetaHuman",
            "Body and eye/teeth anatomy remain native priors; physical scale is uncalibrated",
            "Skin transfer, hair binding, rigging and post-conform likeness review are separate",
        ],
    }
    if selected is not None:
        report["working_baseline"] = {
            "revision": selected["revision"], "manifest": baseline_reference,
            "scene": tracking_scene, "tracking_camera_mode": "fitted_photo" if reference_face_id else "frontal",
            "baseline_scene_modified": False,
        }
    atomic_json(manifest, report)
    if probe["blockers"]:
        return report
    rgba = root / "tracking-rgba8.bin"
    try:
        for key, name in (("tracking_worker", "ailias.head_reconstruction.blender.tracking_view_worker"),
                          ("conform_worker", "ailias.metahuman.head_conforming_worker")):
            report["artifacts"][key] = store.copy_artifact(module_source(name),
                                                           str(root.relative_to(store.root) / "worker"))
        if reference_face_id:
            from ailias.head_reconstruction.geometry.photo_camera import prepare_photo_tracking
            report["photo_tracking"] = prepare_photo_tracking(
                store, source["fit_job"], reference_face_id, root)
            dependencies.extend(report["photo_tracking"]["input_dependencies"])
            atomic_json(manifest, report)
        else:
            tracking = root / "tracking-request.json"
            atomic_json(tracking, {"output": str(root), "size": 512, "device": generation.blender_device,
                                   "camera_mode": "frontal" if selected is not None else "retained"})
            report["tracking_runtime"] = tools.run("blender", "conform-tracking-view", [
                "--background", "--disable-autoexec", store.artifact_path(tracking_scene),
                "--python-exit-code", "1", "--python",
                store.artifact_path(report["artifacts"]["tracking_worker"]), "--", tracking,
            ], root)
        with Image.open(root / "tracking-view.png") as image:
            rgba.write_bytes(image.convert("RGBA").tobytes())
        request = root / "conform-request.json"
        receipt = root / "conform-receipt.json"
        atomic_json(request, {
            "mesh": imported["imported"]["mesh_geometry"][0]["asset"],
            "destination": asset_path(f"/Game/Ailias/ConformedHeads/C{revision}"),
            "camera": str(root / "tracking-camera.json"), "rgba": str(rgba), "receipt": str(receipt),
            "source_character": source_character,
        })
        tools.paths["unreal"] = Path(probe["editor"])
        editor_mode = (["-AllowCommandletRendering", "-RenderOffscreen", "-d3d12"]
                       if source_character else ["-NullRHI"])
        report["conform_runtime"] = tools.run("unreal", "head-only-conform", [
            probe["project"], "-run=pythonscript",
            f"-script={store.artifact_path(report['artifacts']['conform_worker'])}",
            "-unattended", "-nop4", "-nosplash", *editor_mode, "-stdout", "-FullStdOutLogOutput",
            "-ExecCmds=mh.Character.FromCustomMeshImportShowIterations 0",
        ], root, environment={"AILIAS_CONFORM_REQUEST": str(request)})
        result = json.loads(receipt.read_text())
        if result.get("status") != "head_conformed_awaiting_review" or not result.get("assets"):
            raise ValueError("Unreal did not confirm a saved head-only conform")
        before, after = (np.asarray(result[name], dtype=float)
                         for name in ("face_landmarks_before", "face_landmarks_after"))
        if (before.shape != after.shape or before.ndim != 2 or before.shape[1] != 3
                or not before.size or not np.isfinite([before, after]).all()):
            raise ValueError("Native conformed landmark evidence is missing or invalid")
        report["landmark_change_rms"] = float(np.sqrt(np.mean(np.sum((before-after)**2, axis=1))))
        for index, package in enumerate(result["assets"]):
            relative = asset_path(package).removeprefix("/Game/").split(".")[0]
            asset = project.parent / "Content" / Path(*relative.split("/"))
            report["artifacts"][f"character_{index}"] = store.reference_artifact(asset.with_suffix(".uasset"))
        if not all(store.verify(item) for item in dependencies):
            raise ValueError("Imported head changed during conform")
        report.update(status="head_conformed_awaiting_review", character=result["character"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                      project_may_contain_partial_outputs=True)
    finally:
        for name in ("tracking-view.png", "tracking-camera.json", "tracking-request.json",
                     "conform-request.json", "conform-receipt.json"):
            if (root / name).is_file() and (root / name).stat().st_size:
                report["artifacts"][name] = store.reference_artifact(root / name)
        if (root / "conform-receipt.json").is_file():
            try:
                report["native_result"] = json.loads((root / "conform-receipt.json").read_text())
            except (OSError, ValueError):
                report["native_receipt_unreadable"] = True
        finalize_job_resources(store, report, manifest)
        atomic_json(manifest, report)
    return report

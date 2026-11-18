"""Fresh sampled-photo identity fitting and native Character creation, without prior heads."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from PIL import Image

from ailias.facefitting.fitting import prepare_views
from ailias.facefitting.model import HeadModel, model_assets
from ailias.facefitting.photo_tracking import perspective_camera
from ailias.facefitting.pipeline import export_obj, save_fit
from ailias.facefitting.native_features import apply_features, character_package
from ailias.facefitting.native_appearance import bake_native_colour, canonical_to_target_pose
from ailias.facefitting.native_photo_profile import photo_profile
from ailias.facefitting.appearance import load_appearance_policy
from ailias.facefitting.native_jobs import run_native_job
from ailias.facefitting.sampled_photos import load_samples, process_samples
from ailias.facefitting.shared_identity import fit_shared_identity
from ailias.facefitting.video import VideoPolicy
from ailias.generation.unreal import readiness
from ailias.storage import atomic_json
from ailias.reanimation.settings import load_policy as load_review_policy


def create_fresh_video_head(store, evidence_path, editor, project, tools, policy, *,
                            video_policy=None, appearance_policy=None):
    """Run from sampled-photo evidence with fresh identity/cameras/Character; no refinement cache."""
    root = store.resolve(str(Path("facefitting") / "fresh-video" / uuid.uuid4().hex))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    start = time.monotonic()
    report = {
        "revision": root.name, "status": "running", "stages": {}, "artifacts": {},
        "started_at": datetime.now(timezone.utc).isoformat(), "policy": policy.record(),
        "scope": "sampled photos to newly created native Character",
        "previous_head_reused": False, "previous_fit_reused": False,
        "previous_response_reused": False, "previous_appearance_reused": False,
        "blender_used": False, "cloud_requested": False, "likeness_verified": False,
        "input_dependencies": [],
        "sampled_images_reused": True, "previous_predictions_reused": False,
        "video_decoding_performed": False,
    }

    def stage(name, operation):
        report["current_stage"] = name
        report["stages"][name] = {"status": "running"}
        atomic_json(manifest, report)
        tick = time.monotonic()
        try:
            value = operation()
            report["stages"][name]["status"] = "passed"
            return value
        except Exception as error:
            report["stages"][name].update(
                status="failed", error_type=type(error).__name__, error=str(error),
                cause=str(error.__cause__) if error.__cause__ is not None else None)
            raise
        finally:
            report["stages"][name]["seconds"] = round(time.monotonic()-tick, 3)
            atomic_json(manifest, report)
            print(json.dumps({"revision": root.name, "stage": name, **report["stages"][name]}),
                  file=sys.stderr, flush=True)

    try:
        def preflight():
            probe = readiness(editor, project)
            if probe["blockers"]:
                raise ValueError("Unreal prerequisites blocked: "+", ".join(probe["blockers"]))
            return probe

        stage("native_preflight", preflight)
        samples, dependencies = stage("load_sampled_photos", lambda: load_samples(store, evidence_path))
        report["input_dependencies"].extend(dependencies)
        assets = stage("load_models", lambda: model_assets(store))
        report["model_assets"] = assets
        report["input_dependencies"].extend(a["artifact"] for a in assets.values())
        evidence = stage("fresh_photo_processing", lambda: process_samples(
            store, samples, root, assets["landmarker"]["artifact"], video_policy or VideoPolicy()))
        report["photo_features"] = store.reference_artifact(root / "photo-evidence" / "manifest.json")
        report["artifacts"]["photo_features"] = report["photo_features"]
        report["input_dependencies"].extend(
            evidence["parser"][key] for key in ("model", "license_artifact") if key in evidence["parser"])
        model = HeadModel(store.artifact_path(assets["gnm"]["artifact"]),
                          store.artifact_path(assets["correspondence"]["artifact"]), policy.expression_modes)
        observations = [{**frame, "status": "ready"} for frame in evidence["frames"] if "source_xy" in frame]
        entries = {f["face_id"]: {} for f in observations}
        views, excluded = prepare_views(model, observations, entries)
        report["inputs"] = {"images": len(evidence["frames"]), "fit_views": len(views),
                            "excluded": excluded}

        with redirect_stdout(sys.stderr):
            fitted = stage("fresh_shared_identity_fit",
                           lambda: fit_shared_identity(model, views, policy, report))
        vertices = model.neutral(fitted["identity"])
        if not np.isfinite(vertices).all():
            raise ValueError("Fresh identity has nonfinite geometry")
        save_fit(root / "fit.npz", fitted)
        export_obj(root / "neutral-head.obj", vertices, model.triangles)
        # UE's OBJ importer flips handedness on Y, matching the existing Blender export.
        with np.load(store.artifact_path(assets["gnm"]["artifact"]), allow_pickle=False) as gnm:
            triangle_uvs = gnm["triangle_uvs"]
        export_obj(root / "head-for-unreal.obj", vertices[:, [2, 0, 1]]*100,
                   model.triangles, description="fresh fitted head; right-handed OBJ (GNM Z,X,Y), centimetres",
                   triangle_uvs=triangle_uvs)
        expected_native = vertices[:, [2, 0, 1]]*[100, -100, 100]
        report["geometry"] = {
            "identity_modes": len(fitted["identity"]), "vertices": len(vertices),
            "maximum_template_displacement_cm": float(np.linalg.norm(vertices-model.vertices, axis=1).max()*100),
            "physical_scale_calibrated": False,
        }
        by_id = {f["face_id"]: f for f in evidence["frames"]}
        fronts = [i for i, v in enumerate(views) if v["view"]["category"] == "front"]
        if not fronts:
            raise ValueError("Fresh native conform requires a fitted frontal photo")
        index = max(fronts, key=lambda i: by_id[views[i]["face_id"]].get("quality_score", -1e9))
        frame = by_id[views[index]["face_id"]]
        with Image.open(store.artifact_path(frame["image"])) as image:
            tracking = image.convert("RGB")
        tracking.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        tracking.save(root / "tracking-view.png")
        (root / "tracking-rgba8.bin").write_bytes(tracking.convert("RGBA").tobytes())
        camera = perspective_camera(
            {"camera": fitted["per_view"][index, :7], "source_size": frame["source_size"]},
            model.center, tracking.size)
        atomic_json(root / "tracking-camera.json", camera)
        report["tracking_face_id"] = frame["face_id"]
        worker = store.copy_artifact(Path(__file__).with_name("fresh_video_worker.py"),
                                     str(root.relative_to(store.root) / "worker"))
        conform = store.copy_artifact(Path(__file__).with_name("unreal_conform_worker.py"),
                                      str(root.relative_to(store.root) / "worker"))
        report["artifacts"].update(worker=worker, conform_worker=conform)
        tools.paths["unreal"] = Path(editor)

        def native(operation, extra):
            return run_native_job(tools, project, store.artifact_path(worker), root,
                                  operation, extra, report)

        created = stage("fresh_native_conform", lambda: native("create", {
            "obj": str(root / "head-for-unreal.obj"),
            "destination": f"/Game/Ailias/FreshVideo/V{root.name}",
            "camera": str(root / "tracking-camera.json"), "rgba": str(root / "tracking-rgba8.bin"),
            "conform_worker": str(store.artifact_path(conform)),
            "geometry_output": str(root / "native-geometry-before.json"),
            "expected_bounds": [expected_native.min(0).tolist(), expected_native.max(0).tolist()],
        }))
        if created["status"] != "head_conformed_awaiting_review":
            raise ValueError("Fresh native conform did not save a new Character")
        report["character"] = created["character"]
        surface_worker = store.copy_artifact(Path(__file__).with_name("native_surface_worker.py"),
                                            str(root.relative_to(store.root) / "worker"))
        report["artifacts"]["surface_worker"] = surface_worker
        alignment = canonical_to_target_pose(created)
        report["photo_projection_alignment"] = alignment
        verified = stage("independent_native_reload", lambda: native("verify", {
            "character": created["character"], "expected": created["face_landmarks_saved"],
            "expected_surface": str(root / "native-geometry-before.json"),
            "surface_destination": f"/Game/Ailias/FreshVideo/V{root.name}/ReloadedSurface",
            "surface_worker": str(store.artifact_path(surface_worker)),
            "native_uv_output": str(root / "native-skin-uvs.json"),
            "canonical_to_target": alignment,
            "conform_worker": str(store.artifact_path(conform))}))
        if verified["status"] != "fresh_character_reloaded":
            raise ValueError("Fresh Character reload failed")
        report["artifacts"]["character"] = store.reference_artifact(
            character_package(project, created["character"]))
        appearance = appearance_policy if appearance_policy is not None else load_appearance_policy()
        report["appearance_policy"] = appearance
        preview_worker = store.copy_artifact(Path(__file__).with_name("native_preview_worker.py"),
                                            str(root.relative_to(store.root) / "worker"))
        report["artifacts"].update(surface_worker=surface_worker, preview_worker=preview_worker)

        def surface_job(operation, parameters):
            return run_native_job(tools, project, store.artifact_path(surface_worker), root,
                                  operation, parameters, report)

        report["native_skin_uvs"] = verified["native_skin_uvs"]
        baked = stage("bake_fresh_photo_colour", lambda: bake_native_colour(
            store, root, root / "native-skin-uvs.json", model, fitted, views, evidence,
            appearance["device"], texture_size=appearance["texture_size"],
            maximum_views=appearance["max_views"], photo_size=appearance["photo_size"]))
        report["photo_colour"] = baked
        imported = stage("import_native_photo_texture", lambda: surface_job("import_texture", {
            "image": baked["texture"], "destination": f"/Game/Ailias/FreshVideo/V{root.name}/PhotoTexture"}))

        def build_profile():
            return photo_profile(evidence, baked, imported["texture"], [
                report["photo_features"], store.reference_artifact(Path(baked["texture"])),
                store.reference_artifact(character_package(project, imported["texture"]))])

        profile, mapping = stage("derive_native_appearance", build_profile)
        report["appearance_mapping"] = mapping
        profile_path = root / "photo-native-profile.json"
        atomic_json(profile_path, profile)

        def apply_native():
            result = apply_features(store, created["character"], profile_path, editor, project, tools,
                                    use_working_baseline=False)
            report["native_appearance"] = result
            if result["status"] != "features_reloaded_awaiting_visual_review":
                raise RuntimeError("Fresh native appearance did not persist: " + str(result.get("error", result["status"])))
            return result

        applied = stage("apply_and_reload_native_appearance", apply_native)
        report["character"] = applied["character"]
        report["artifacts"]["character"] = applied["artifacts"]["character"]
        preview = stage("render_new_native_preview", lambda: run_native_job(
            tools, project, store.artifact_path(preview_worker), root, "preview",
            {"character": applied["character"], "destination": f"/Game/Ailias/FreshVideo/V{root.name}/Preview",
             "output": str(root / "native-preview.png"), "warmup_seconds": 20,
             "lighting": load_review_policy().to_dict()}, report))
        with Image.open(root / "native-preview.png") as image:
            if image.size != (540, 960) or np.asarray(image.convert("RGB")).std() < 2:
                raise RuntimeError("Native preview is empty or has unexpected dimensions")
        report["preview"] = preview
        if not all(store.verify(item) for item in report["input_dependencies"]):
            raise ValueError("Fresh inputs changed during execution")
        report.update(
            status="fresh_video_completed", current_stage="awaiting_visual_review",
            fresh_editor_reload_verified=True, assembled=False,
            limitations=mapping["limitations"] + [
                "Editor-native preview is not an assembled animation rig",
                "Successful automation and save/reload do not establish personal likeness"])
    except Exception as error:
        report.update(status="failed", error={
            "type": type(error).__name__, "message": str(error),
            "cause": str(error.__cause__) if error.__cause__ is not None else None,
            "notes": getattr(error, "__notes__", []),
        }, project_may_contain_partial_outputs="fresh_native_conform" in report["stages"])
    finally:
        for path in root.iterdir():
            try:
                if path.is_file() and path != manifest and path.stat().st_size:
                    report["artifacts"][path.name] = store.reference_artifact(path)
            except OSError as error:
                report.setdefault("artifact_errors", []).append({
                    "file": path.name, "type": type(error).__name__})
        if report.get("artifact_errors"):
            report["status"] = "failed"
        report["seconds"] = round(time.monotonic()-start, 3)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest, report)
    return report

"""Fresh sampled-photo identity fitting and native Character creation, without prior heads."""

from contextlib import redirect_stdout
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from PIL import Image

from ailias.facefitting.appearance.profile_mapping import build_appearance_profile
from ailias.facefitting.blender.appearance_scene import create_blender_head
from ailias.facefitting.appearance.skin_texture_projection import (
    bake_photo_skin_texture,
    estimate_photo_alignment,
)
from ailias.facefitting.configuration import DEFAULT_REVIEW_CONFIG, VideoPolicy, load_appearance_policy
from ailias.facefitting.evidence.photo_processing import (
    extract_photo_evidence,
    load_retained_frames,
)
from ailias.facefitting.evidence.reviewed_photos import prepare_reviewed_photos
from ailias.facefitting.evidence.fit_comparison import write_fit_overlays
from ailias.facefitting.geometry.artifact_io import export_obj, save_fit
from ailias.facefitting.geometry.head_model import HeadModel, model_assets
from ailias.facefitting.geometry.identity_fitting import fit_multiview_identity
from ailias.facefitting.geometry.identity_search import search_identity_fit
from ailias.facefitting.geometry.optimization import prepare_views
from ailias.facefitting.geometry.photo_camera import perspective_camera
from ailias.facefitting.unreal.character_features import apply_character_features, character_package
from ailias.facefitting.unreal.worker_process import execute_unreal_worker
from ailias.generation.unreal import readiness
from ailias.generation.settings import load_settings
from ailias.storage import atomic_json
from ailias.imageprocessing.pipeline import read_image
from ailias.reanimation.settings import load_policy as load_review_policy


def run_photo_to_character_pipeline(
    store, evidence_path, editor, project, tools, policy, *,
    video_policy=None, appearance_policy=None, generation_policy=None,
    photos_run=None, search_policy=None, review_policy=None,
):
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

    def emit_progress(event):
        try:
            print(json.dumps(event), file=sys.stderr, flush=True)
        except OSError:
            pass

    def run_stage(name, operation):
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
            emit_progress({"revision": root.name, "stage": name, **report["stages"][name]})

    try:
        def check_unreal_prerequisites():
            probe = readiness(editor, project)
            if probe["blockers"]:
                raise ValueError("Unreal prerequisites blocked: "+", ".join(probe["blockers"]))
            return probe

        run_stage("native_preflight", check_unreal_prerequisites)
        samples, dependencies = run_stage("load_sampled_photos", lambda: load_retained_frames(store, evidence_path))
        report["input_dependencies"].extend(dependencies)
        if photos_run is not None:
            photos, photo_dependencies = run_stage("load_reviewed_original_photos", lambda: prepare_reviewed_photos(
                store, photos_run, root))
            samples.extend(photos)
            report["input_dependencies"].extend(photo_dependencies)
            report["artifacts"]["original_photo_inputs"] = store.reference_artifact(root / "photo-inputs" / "manifest.json")
        assets = run_stage("load_models", lambda: model_assets(store))
        report["model_assets"] = assets
        report["input_dependencies"].extend(a["artifact"] for a in assets.values())
        evidence = run_stage("fresh_photo_processing", lambda: extract_photo_evidence(
            store, samples, root, assets["landmarker"]["artifact"], video_policy or VideoPolicy()))
        report["photo_features"] = store.reference_artifact(root / "photo-evidence" / "manifest.json")
        report["artifacts"]["photo_features"] = report["photo_features"]
        report["input_dependencies"].extend(
            evidence["parser"][key] for key in ("model", "license_artifact") if key in evidence["parser"])
        model = HeadModel(store.artifact_path(assets["gnm"]["artifact"]),
                          store.artifact_path(assets["correspondence"]["artifact"]), policy.expression_modes)
        report["model"] = model.audit()
        observations = [{**frame, "status": "ready"} for frame in evidence["frames"] if "source_xy" in frame]
        entries = {f["face_id"]: {} for f in observations}
        views, excluded = prepare_views(model, observations, entries)
        report["inputs"] = {"images": len(evidence["frames"]), "fit_views": len(views),
                            "excluded": excluded,
                            "strict_neutral_views": sum(frame.get("eligible_identity", False)
                                                        for frame in evidence["frames"]),
                            "quality_warnings": dict(Counter(
                                reason for frame in evidence["frames"] for reason in frame.get("reasons", [])))}

        with redirect_stdout(sys.stderr):
            if search_policy is None:
                fitted = run_stage("fresh_shared_identity_fit",
                                   lambda: fit_multiview_identity(model, views, policy, report))
            else:
                candidates = root / "fit-search"
                candidates.mkdir()

                def retain_candidate(record, candidate):
                    name = f"candidate-{record['index']:03d}"
                    if candidate is not None:
                        path = candidates / (name + ".npz")
                        save_fit(path, candidate)
                        report["artifacts"][name] = store.reference_artifact(path)
                    atomic_json(candidates / (name + ".json"), record)
                    atomic_json(manifest, report)
                    emit_progress({"candidate": record["index"], "status": record["status"],
                                   "identity_modes": record["identity_modes"],
                                   "expression_modes": record["policy"]["expression_modes"],
                                   "score": record.get("score")})

                report["requested_policy"] = policy.record()
                model, fitted, policy = run_stage("search_shared_identity_fit", lambda: search_identity_fit(
                    lambda modes: HeadModel(store.artifact_path(assets["gnm"]["artifact"]),
                                            store.artifact_path(assets["correspondence"]["artifact"]), modes),
                    views, policy, search_policy, report, retain_candidate))
                report["policy"], report["model"] = policy.record(), model.audit()
        report["fit"] = fitted["report"]
        if search_policy is not None:
            report["artifacts"].update(run_stage(
                "write_fit_comparisons", lambda: write_fit_overlays(store, root, views, fitted, evidence)))
        vertices = model.neutral(fitted["identity"])
        if not np.isfinite(vertices).all():
            raise ValueError("Fresh identity has nonfinite geometry")
        save_fit(root / "fit.npz", fitted)
        export_obj(root / "neutral-head.obj", vertices, model.triangles)
        # UE's OBJ importer flips handedness on Y, matching the existing Blender export.
        with np.load(store.artifact_path(assets["gnm"]["artifact"]), allow_pickle=False) as gnm:
            triangle_uvs = gnm["triangle_uvs"].copy()
        appearance = appearance_policy if appearance_policy is not None else load_appearance_policy()
        generation = generation_policy if generation_policy is not None else load_settings()[0]
        report["appearance_policy"] = appearance
        report["blender_manifest"] = str(root / "blender" / "manifest.json")
        blender = run_stage("create_blender_appearance", lambda: create_blender_head(
            store, root / "blender", model, fitted, views, evidence, triangle_uvs, tools,
            generation, appearance,
        ))
        report["blender"] = blender
        report["blender_used"] = True
        report["artifacts"].update({"blender_" + key: value
                                     for key, value in blender["artifacts"].items()})
        unreal_mesh = store.artifact_path(blender["artifacts"]["head_only_mesh"])
        expected_native = vertices[:, [2, 0, 1]]*[100, -100, 100]
        displacement = np.linalg.norm(vertices-model.vertices, axis=1)
        report["geometry"] = {
            "identity_modes": len(fitted["identity"]), "vertices": len(vertices),
            "maximum_template_displacement_cm": float(displacement.max()*100),
            "median_template_displacement_cm": float(np.median(displacement)*100),
            "p95_template_displacement_cm": float(np.quantile(displacement, .95)*100),
            "vertices_moved_over_1mm_model_units": int((displacement > .001).sum()),
            "constraints": "shared identity basis fitted to supported 2D landmark correspondences; not dense measured vertices",
            "physical_scale_calibrated": False,
        }
        by_id = {f["face_id"]: f for f in evidence["frames"]}
        fronts = [i for i, v in enumerate(views) if v["view"]["category"] == "front"]
        if not fronts:
            raise ValueError("Fresh native conform requires a fitted frontal photo")
        index = max(fronts, key=lambda i: by_id[views[i]["face_id"]].get("quality_score", -1e9))
        frame = by_id[views[index]["face_id"]]
        tracking, _ = read_image(store.artifact_path(frame["image"]))
        with tracking:
            tracking.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            tracking.save(root / "tracking-view.png")
            with tracking.convert("RGBA") as rgba:
                (root / "tracking-rgba8.bin").write_bytes(rgba.tobytes())
            camera = perspective_camera(
                {"camera": fitted["per_view"][index, :7], "source_size": frame["source_size"]},
                model.center, tracking.size)
        atomic_json(root / "tracking-camera.json", camera)
        report["tracking_face_id"] = frame["face_id"]
        worker = store.copy_artifact(
            Path(__file__).parents[1] / "unreal" / "character_creation_worker.py",
            str(root.relative_to(store.root) / "worker"),
        )
        conform = store.copy_artifact(
            Path(__file__).parents[1] / "unreal" / "head_conforming_worker.py",
            str(root.relative_to(store.root) / "worker"),
        )
        report["artifacts"].update(worker=worker, conform_worker=conform)
        tools.paths["unreal"] = Path(editor)

        def execute_character_operation(operation, extra):
            return execute_unreal_worker(
                tools, project, store.artifact_path(worker), root, operation, extra, report,
            )

        created = run_stage("fresh_native_conform", lambda: execute_character_operation("create", {
            "obj": str(unreal_mesh),
            "destination": f"/Game/Ailias/FreshVideo/V{root.name}",
            "camera": str(root / "tracking-camera.json"), "rgba": str(root / "tracking-rgba8.bin"),
            "conform_worker": str(store.artifact_path(conform)),
            "geometry_output": str(root / "native-geometry-before.json"),
            "expected_bounds": [expected_native.min(0).tolist(), expected_native.max(0).tolist()],
        }))
        if created["status"] != "head_conformed_awaiting_review":
            raise ValueError("Fresh native conform did not save a new Character")
        report["character"] = created["character"]
        surface_worker = store.copy_artifact(
            Path(__file__).parents[1] / "unreal" / "skin_texture_worker.py",
            str(root.relative_to(store.root) / "worker"),
        )
        report["artifacts"]["surface_worker"] = surface_worker
        alignment = estimate_photo_alignment(created)
        report["photo_projection_alignment"] = alignment
        verified = run_stage("independent_native_reload", lambda: execute_character_operation("verify", {
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
        preview_worker = store.copy_artifact(
            Path(__file__).parents[1] / "unreal" / "character_portrait_worker.py",
            str(root.relative_to(store.root) / "worker"),
        )
        report["artifacts"].update(surface_worker=surface_worker, preview_worker=preview_worker)

        def execute_texture_operation(operation, parameters):
            return execute_unreal_worker(
                tools, project, store.artifact_path(surface_worker), root, operation, parameters, report,
            )

        report["native_skin_uvs"] = verified["native_skin_uvs"]
        baked = run_stage("bake_fresh_photo_colour", lambda: bake_photo_skin_texture(
            store, root, root / "native-skin-uvs.json", model, fitted, views, evidence,
            appearance["device"], texture_size=appearance["texture_size"],
            maximum_views=appearance["max_views"], photo_size=appearance["photo_size"]))
        report["photo_colour"] = baked
        imported = run_stage("import_native_photo_texture", lambda: execute_texture_operation("import_texture", {
            "image": baked["texture"], "destination": f"/Game/Ailias/FreshVideo/V{root.name}/PhotoTexture"}))

        def map_appearance_evidence():
            return build_appearance_profile(evidence, baked, imported["texture"], [
                report["photo_features"], store.reference_artifact(Path(baked["texture"])),
                store.reference_artifact(character_package(project, imported["texture"]))],
                native_hair_asset=appearance.get("native_hair_asset"))

        profile, mapping = run_stage("derive_native_appearance", map_appearance_evidence)
        report["appearance_mapping"] = mapping
        profile_path = root / "photo-native-profile.json"
        atomic_json(profile_path, profile)

        def apply_appearance_profile():
            result = apply_character_features(
                store, created["character"], profile_path, editor, project, tools,
                use_working_baseline=False,
            )
            report["native_appearance"] = result
            if result["status"] != "features_reloaded_awaiting_visual_review":
                raise RuntimeError("Fresh native appearance did not persist: " + str(result.get("error", result["status"])))
            return result

        applied = run_stage("apply_and_reload_native_appearance", apply_appearance_profile)
        report["character"] = applied["character"]
        report["artifacts"]["character"] = applied["artifacts"]["character"]
        report["review_policy"] = (
            review_policy if review_policy is not None else load_review_policy(DEFAULT_REVIEW_CONFIG)
        ).to_dict()
        preview = run_stage("render_new_native_preview", lambda: execute_unreal_worker(
            tools, project, store.artifact_path(preview_worker), root, "preview",
            {"character": applied["character"], "destination": f"/Game/Ailias/FreshVideo/V{root.name}/Preview",
             "output": str(root / "native-preview.png"), "warmup_seconds": 20,
             "lighting": report["review_policy"]}, report))
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
        blender_manifest = root / "blender" / "manifest.json"
        if blender_manifest.is_file():
            report["artifacts"]["blender_manifest"] = store.reference_artifact(blender_manifest)
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

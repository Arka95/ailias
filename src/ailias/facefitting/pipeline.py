"""Orchestrate fresh photo-conditioned fits, quality gates and neutral review outputs.

Retains fit/source provenance without completing manual likeness or MetaHuman stages.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ailias.facefitting.fitting import evaluation_score, fit, prepare_views, validation_indices
from ailias.facefitting.model import HeadModel, model_assets
from ailias.facefitting.observations import DenseDetector, observe
from ailias.generation.inputs import prepare_inputs
from ailias.imageprocessing.curation import load_filter
from ailias.imageprocessing.pipeline import component, load_run
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.resources import finalize_resources
from ailias.storage import atomic_json


def export_obj(path, vertices, triangles):
    with path.open("w", encoding="ascii") as stream:
        stream.write("# ailias photo-conditioned neutral GNM head; uncalibrated scale\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b, c in triangles + 1:
            stream.write(f"f {a} {b} {c}\n")


def save_fit(path, result):
    np.savez_compressed(path, identity=result["identity"], per_view=result["per_view"],
                        parameters=result["parameters"], projected=result["projected"])


def overlay(store, root, observation, projected, weights, name):
    with Image.open(store.artifact_path(observation["aligned_cutout"])) as image:
        image = image.convert("RGB")
    size = np.asarray(observation["source_size"])
    source = projected * max(size) + size / 2
    matrix = np.asarray(observation["alignment"]["source_to_aligned"])
    predicted = (np.c_[source, np.ones(len(source))] @ matrix.T)[:, :2]
    draw = ImageDraw.Draw(image)
    observed = np.asarray(observation["aligned_xy"])[:468]
    for i in np.flatnonzero(weights):
        x, y = observed[i]
        a, b = predicted[i]
        draw.line((x, y, a, b), fill=(240, 190, 30))
        draw.ellipse((x-1, y-1, x+1, y+1), fill=(60, 170, 255))
        draw.ellipse((a-1, b-1, a+1, b+1), fill=(40, 255, 110))
    draw.text((8, 8), "Blue: predicted image landmarks / Green: fitted projection", fill="white")
    path = root / name
    image.save(path)
    return store.reference_artifact(path)


def render_review(store, root, model, vertices, tools, generation_policy):
    geometry = root / "review-geometry.npz"
    np.savez_compressed(geometry, prior=model.vertices, fitted=vertices, triangles=model.triangles,
                        eyes=model.groups["eyes"], pupils=model.groups["pupils"],
                        irises=model.groups["irises"], teeth=model.groups["teeth"])
    source_worker = Path(__file__).with_name("blender_worker.py")
    worker_artifact = store.copy_artifact(source_worker, str(root.relative_to(store.root) / "worker"))
    worker = store.artifact_path(worker_artifact)
    request = root / "render-request.json"
    atomic_json(request, {"geometry": str(geometry), "output": str(root),
                          "device": generation_policy.blender_device,
                          "size": generation_policy.render_size,
                          "samples": generation_policy.render_samples})
    result = tools.run("blender", "head-review",
                       ["--background", "--factory-startup", "--python-exit-code", "1",
                        "--python", str(worker), "--", str(request)], root)
    artifacts = {"scene": store.reference_artifact(root / "head-review.blend"),
                 "render_request": store.reference_artifact(request),
                 "render_worker": worker_artifact,
                 "render_geometry": store.reference_artifact(geometry)}
    artifacts["render_runtime"] = store.reference_artifact(root / "render-runtime.json")
    for label in ("prior", "fitted"):
        for angle in (0, 45, 90):
            name = f"{label}-{angle}.png"
            with Image.open(root / name) as image:
                image.verify()
            artifacts[name] = store.reference_artifact(root / name)
    side = generation_policy.render_size
    sheet = Image.new("RGB", (side*3, (side+28)*2), "#181818")
    draw = ImageDraw.Draw(sheet)
    for row, label in enumerate(("prior", "fitted")):
        for column, angle in enumerate((0, 45, 90)):
            with Image.open(root / f"{label}-{angle}.png") as image:
                sheet.paste(image.convert("RGB"), (column*side, row*(side+28)+28))
            draw.text((column*side+8, row*(side+28)+8), f"{label.upper()} / {angle} degrees", fill="white")
    sheet.save(root / "head-comparison.jpg", quality=94)
    artifacts["comparison"] = store.reference_artifact(root / "head-comparison.jpg")
    return artifacts, result


def run_fit(store, run, group, policy, generation_policy, tools):
    started = time.monotonic()
    inputs = prepare_inputs(store, run, group, generation_policy)
    assets = model_assets(store)
    extraction = load_run(store, run)
    entries = {entry["face_id"]: entry for entry in inputs["inputs"]["entries"]}
    job = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = store.resolve(str(Path("facefitting") / "runs" / job))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    report = {
        "schema_version": 1, "job": job, "run": run, "group": group,
        "status": "running", "current_step": "dense_observations",
        "policy": policy.record(), "generation_policy": generation_policy.__dict__,
        "inputs": store.reference_artifact(store.resolve(inputs["manifest"])),
        "assets": assets, "artifacts": {}, "candidates": [],
        "limitations": [
            "Image landmarks and facial depth are network predictions, not independent measurements",
            "Fitted scale and focal length remain uncalibrated; no millimetre accuracy claim",
            "Front-facing normals/alpha are occlusion heuristics; glasses and hidden features remain uncertain",
            "No fitted hair, intrinsic skin albedo, personalized ears/rear head, rig or MetaHuman conversion",
            "Held-out-view reprojection is conditional on fitted camera/expression, not ground-truth 3D likeness",
        ],
    }
    atomic_json(manifest, report)
    try:
        model = HeadModel(store.artifact_path(assets["gnm"]["artifact"]),
                          store.artifact_path(assets["correspondence"]["artifact"]), policy.expression_modes)
        report["model_audit"] = model.audit()
        detector = DenseDetector(store.artifact_path(assets["landmarker"]["artifact"]))
        observations = []
        try:
            for entry in entries.values():
                metadata = extraction["sources"][entry["source_id"]]["metadata"]
                observations.append(observe(store, entry, metadata["oriented_size"], detector))
        finally:
            detector.close()
        observed_path = root / "observations.json"
        atomic_json(observed_path, {"method": "mediapipe-478-reversible-source-v1", "faces": observations})
        report["artifacts"]["observations"] = store.reference_artifact(observed_path)
        views, excluded = prepare_views(model, observations, entries)
        report["observation_counts"] = {
            "inputs": len(entries), "dense_ready": sum(o["status"] == "ready" for o in observations),
            "fit_eligible": len(views), "excluded": excluded,
        }
        heldout = validation_indices(views)
        training = [i for i in range(len(views)) if i not in heldout]
        train_views, validation_views = ([views[i] for i in indices] for indices in (training, heldout))
        report["split"] = {
            "method": "every fifth face-ID-sorted observation within each dense view category",
            "training": [v["face_id"] for v in train_views],
            "heldout_identity": [v["face_id"] for v in validation_views],
            "final_refit": "All eligible views after candidate selection; final residual is not held-out validation",
        }
        report["current_step"] = "fit_generic_shape_reference"
        atomic_json(manifest, report)
        baseline_train = fit(model, train_views, policy, 0, fixed_identity=np.zeros(0))
        baseline_validation = fit(model, validation_views, policy, 0, fixed_identity=np.zeros(0))
        report["generic_reference"] = {"training": baseline_train["report"],
                                       "heldout_identity": baseline_validation["report"]}
        baseline_evaluation = evaluation_score(baseline_validation["report"], validation_views)
        best = None
        for modes in policy.identity_candidates:
            report["current_step"] = f"fit_{modes}_identity_modes"
            atomic_json(manifest, report)
            initial = np.r_[np.zeros(modes), baseline_train["per_view"].ravel()]
            candidate = fit(model, train_views, policy, modes, initial=initial)
            validation = fit(model, validation_views, policy, modes, fixed_identity=candidate["identity"],
                             initial=baseline_validation["per_view"].ravel())
            evaluation = evaluation_score(validation["report"], validation_views)
            score = evaluation["view_balanced_mean_error"]
            front = evaluation["per_initial_view_category"].get("front")
            baseline_front = baseline_evaluation["per_initial_view_category"].get("front")
            frontal_preserved = front is None or baseline_front is None or front <= baseline_front * 1.05
            path = root / f"candidate-{modes}.npz"
            save_fit(path, candidate)
            report["artifacts"][f"candidate_{modes}"] = store.reference_artifact(path)
            report["candidates"].append({"identity_modes": modes, "training": candidate["report"],
                                         "heldout_identity": validation["report"],
                                         "evaluation": evaluation, "frontal_preserved": frontal_preserved})
            if frontal_preserved and (best is None or score < best[0]):
                best = score, candidate, validation, evaluation
        if best is None:
            raise ValueError("No candidate preserves held-out frontal accuracy within 5%")
        score, candidate, validation, evaluation = best
        baseline_score = baseline_evaluation["view_balanced_mean_error"]
        report["selection"] = {
            "identity_modes": len(candidate["identity"]),
            "criterion": "lowest view-category-balanced mean held-out landmark error, "
                         "at least 1% improvement and no more than 5% frontal regression",
            "baseline_heldout_error": baseline_score, "selected_heldout_error": score,
            "relative_heldout_improvement": 1 - score / max(baseline_score, 1e-12),
            "baseline_evaluation": baseline_evaluation, "selected_evaluation": evaluation,
        }
        if score >= baseline_score * .99:
            raise ValueError("Shared identity did not improve held-out views by 1%; no personalized head published")
        report["current_step"] = "refit_selected_shape_on_all_eligible_views"
        atomic_json(manifest, report)
        per_view = np.empty((len(views), candidate["per_view"].shape[1]))
        per_view[training] = candidate["per_view"]
        per_view[heldout] = validation["per_view"]
        result = fit(model, views, policy, len(candidate["identity"]),
                     initial=np.r_[candidate["identity"], per_view.ravel()])
        report["fit"] = result["report"]
        vertices = model.neutral(result["identity"])
        if not np.isfinite(vertices).all():
            raise ValueError("Non-finite neutral head")
        template_edges = model.vertices[model.triangles[:, 1:]] - model.vertices[model.triangles[:, :1]]
        fitted_edges = vertices[model.triangles[:, 1:]] - vertices[model.triangles[:, :1]]
        template_area = np.linalg.norm(np.cross(template_edges[:, 0], template_edges[:, 1]), axis=1)
        fitted_area = np.linalg.norm(np.cross(fitted_edges[:, 0], fitted_edges[:, 1]), axis=1)
        area_ratio = fitted_area / np.maximum(template_area, 1e-14)
        report["mesh_quality"] = {
            "finite": True, "minimum_triangle_double_area": float(fitted_area.min()),
            "triangle_area_ratio_quantiles": np.quantile(area_ratio, [0, .01, .5, .99, 1]).tolist(),
            "self_intersections_checked": False,
        }
        if np.any(fitted_area <= 1e-14) or np.quantile(area_ratio, .01) < .1 or np.quantile(area_ratio, .99) > 10:
            raise ValueError("Head deformation collapses or excessively stretches the template")
        save_fit(root / "fit.npz", result)
        export_obj(root / "neutral-head.obj", vertices, model.triangles)
        report["artifacts"].update(
            mesh=store.reference_artifact(root / "neutral-head.obj"),
            parameters=store.reference_artifact(root / "fit.npz"))
        camera_path = root / "cameras.json"
        atomic_json(camera_path, {
            "model_center": model.center.tolist(),
            "projection": "Rodrigues(world - model_center) + translation; exp(log_focal)*xy/z; "
                          "source_xy = normalized_xy*max(source_size) + source_size/2",
            "expression_names": model.expression_names,
            "identity_names": model.identity_names[:len(result["identity"])],
            "identity": result["identity"].tolist(),
            "views": [{**metric, "camera": result["per_view"][i, :7].tolist(),
                       "expression": result["per_view"][i, 7:].tolist(),
                       "source_size": views[i]["source_size"].tolist(),
                       "landmark_weights": views[i]["weights"].tolist()}
                      for i, metric in enumerate(result["report"]["views"])],
        })
        report["artifacts"]["cameras"] = store.reference_artifact(camera_path)
        observed_by_id = {o["face_id"]: o for o in observations}
        overlay_paths = []
        for i, view in enumerate(views):
            artifact = overlay(store, root, observed_by_id[view["face_id"]], result["projected"][i],
                               view["weights"], f"overlay-{i:02}.png")
            report["artifacts"][f"overlay_{i:02}"] = artifact
            overlay_paths.append(store.artifact_path(artifact))
        columns, thumb = 8, 192
        preview = Image.new("RGB", (columns*thumb, ((len(views)+columns-1)//columns)*(thumb+24)), "#181818")
        draw = ImageDraw.Draw(preview)
        for i, path in enumerate(overlay_paths):
            x, y = i % columns * thumb, i // columns * (thumb+24)
            with Image.open(path) as image:
                preview.paste(image.resize((thumb, thumb)), (x, y))
            source = Path(entries[views[i]["face_id"]]["source"]["path"]).stem
            draw.text((x+3, y+thumb+3), source[:29], fill="white")
        preview.save(root / "landmark-fit-preview.jpg", quality=93)
        report["artifacts"]["landmark_preview"] = store.reference_artifact(root / "landmark-fit-preview.jpg")
        report["current_step"] = "render_prior_and_fitted_head"
        atomic_json(manifest, report)
        artifacts, render = render_review(store, root, model, vertices, tools, generation_policy)
        report["artifacts"].update(artifacts)
        report["render"] = render
        gallery = load_filter(store, extraction["backend"], ProcessingPolicy(**extraction["policy"]))
        if gallery is None or gallery.fingerprint != inputs["inputs"]["curation_sha256"]:
            raise ValueError("Portrait review changed during fitting; generated result is stale")
        dependencies = [entry["source"] for entry in entries.values()]
        dependencies += [observation["aligned_cutout"] for observation in observations]
        if not all(store.verify(artifact) for artifact in dependencies):
            raise ValueError("Photo/analysis input changed during fitting")
        if not all(store.verify(asset["artifact"]) for asset in assets.values()):
            raise ValueError("Head model assets changed during fitting")
        report.update(status="awaiting_human_likeness_review", current_step="review_neutral_mesh_and_overlays",
                      input_dependencies=dependencies)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        report["total_seconds"] = round(time.monotonic()-started, 3)
        finalize_resources(store, report, manifest)
        atomic_json(manifest, report)
    return report


def inspect_fit(store, job):
    path = store.resolve(str(Path("facefitting") / "runs" / component(job) / "manifest.json"))
    report = json.loads(path.read_text(encoding="utf-8"))
    artifacts = [report["inputs"], *report["artifacts"].values(), *report.get("input_dependencies", [])]
    artifacts += [asset["artifact"] for asset in report["assets"].values()]
    problems = [artifact["path"] for artifact in artifacts if not store.verify(artifact)]
    return {**report, "integrity_problems": problems}

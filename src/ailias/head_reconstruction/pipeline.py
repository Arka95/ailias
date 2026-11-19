"""Orchestrate fresh photo-conditioned fits, quality gates and neutral review outputs.

Retains fit/source provenance without completing manual likeness or MetaHuman stages.
"""

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ailias.head_reconstruction.geometry.optimization import prepare_views
from ailias.head_reconstruction.geometry.identity_fitting import (
    fit_shared_head_identity,
)
from ailias.head_reconstruction.geometry.feature_reconstruction import (
    construct_neutral_head_mesh,
)
from ailias.head_reconstruction.reconstruction_data import (
    ExcludedHeadView,
    HeadFittingResult,
    ReconstructionFailure,
    ReconstructionInputs,
    ReconstructionReport,
)
from ailias.head_reconstruction.geometry.head_model import HeadModel, model_assets
from ailias.imageprocessing.dense import DenseDetector, observe
from ailias.head_reconstruction.blender.clay_rendering import render_clay_portrait
from ailias.generation.inputs import prepare_inputs
from ailias.imageprocessing.curation import load_filter
from ailias.imageprocessing.pipeline import load_run
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.resource_management import finalize_job_resources
from ailias.head_reconstruction.geometry.artifact_io import export_obj, save_fit
from ailias.storage import atomic_json


def write_landmark_fit_overlay(store, root, observation, projected, weights, name):
    """Draw observed and fitted landmarks on the existing aligned review image.
    Convert source-grid projections with its retained reversible transform.
    """
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
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(60, 170, 255))
        draw.ellipse((a - 1, b - 1, a + 1, b + 1), fill=(40, 255, 110))
    draw.text(
        (8, 8),
        "Blue: predicted image landmarks / Green: fitted projection",
        fill="white",
    )
    path = root / name
    image.save(path)
    return store.reference_artifact(path)


def collect_photo_landmark_observations(store, entries, extraction, model_path):
    """Extract observations from verified prepared photos with one task-scoped detector.
    Keep the existing source/alignment metadata needed by retained review artifacts.
    """
    detector = DenseDetector(model_path)
    try:
        return [
            observe(
                store,
                entry,
                extraction["sources"][entry["source_id"]]["metadata"]["oriented_size"],
                detector,
            )
            for entry in entries.values()
        ]
    finally:
        detector.close()


def save_fitted_photo_cameras(path, model, fitted: HeadFittingResult, views):
    """Export typed parameters on the existing source-pixel camera contract.
    Retain face IDs and view labels required by saved photo-tracking/appearance consumers.
    """
    atomic_json(
        path,
        {
            "model_center": model.center.tolist(),
            "projection": "Rodrigues(world - model_center) + translation; exp(log_focal)*xy/z; "
            "source_xy = normalized_xy*max(source_size) + source_size/2",
            "expression_names": model.expression_names,
            "identity_names": model.identity_names[
                : len(fitted.parameters.identity_coefficients)
            ],
            "identity": fitted.parameters.identity_coefficients.tolist(),
            "views": [
                {
                    **metric.to_dict(),
                    "face_id": metric.view_id,
                    "view": metric.orientation,
                    "camera": fitted.parameters.cameras_and_expressions[
                        index, :7
                    ].tolist(),
                    "expression": fitted.parameters.cameras_and_expressions[
                        index, 7:
                    ].tolist(),
                    "source_size": views[index]["source_size"].tolist(),
                    "landmark_weights": views[index]["weights"].tolist(),
                }
                for index, metric in enumerate(fitted.summary.views)
            ],
        },
    )


def write_photo_fit_previews(
    store, root, entries, observations, views, fitted: HeadFittingResult
):
    """Save per-view residual overlays and their compact review contact sheet.
    Keep all filenames and source labels local to this fresh fitting job.
    """
    observed_by_id = {
        observation["face_id"]: observation for observation in observations
    }
    artifacts, overlay_paths = {}, []
    for index, view in enumerate(views):
        artifact = write_landmark_fit_overlay(
            store,
            root,
            observed_by_id[view["face_id"]],
            fitted.parameters.projected_landmarks[index],
            view["weights"],
            f"overlay-{index:02}.png",
        )
        artifacts[f"overlay_{index:02}"] = artifact
        overlay_paths.append(store.artifact_path(artifact))
    columns, thumb = 8, 192
    preview = Image.new(
        "RGB",
        (columns * thumb, ((len(views) + columns - 1) // columns) * (thumb + 24)),
        "#181818",
    )
    draw = ImageDraw.Draw(preview)
    try:
        for index, path in enumerate(overlay_paths):
            x, y = index % columns * thumb, index // columns * (thumb + 24)
            with Image.open(path) as image:
                preview.paste(image.resize((thumb, thumb)), (x, y))
            source = Path(entries[views[index]["face_id"]]["source"]["path"]).stem
            draw.text((x + 3, y + thumb + 3), source[:29], fill="white")
        preview.save(root / "landmark-fit-preview.jpg", quality=93)
    finally:
        preview.close()
    artifacts["landmark_preview"] = store.reference_artifact(
        root / "landmark-fit-preview.jpg"
    )
    return artifacts


def fit_head_from_photos(store, run, group, policy, generation_policy, tools):
    """Orchestrate one retained-photo job through shared typed fitting and explicit exports.
    Preserve source integrity/review gates without duplicating numerical identity or mesh checks.
    """
    started = time.monotonic()
    inputs = prepare_inputs(store, run, group, generation_policy)
    assets = model_assets(store)
    extraction = load_run(store, run)
    entries = {entry["face_id"]: entry for entry in inputs["inputs"]["entries"]}
    job = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    root = store.resolve(str(Path("facefitting") / "runs" / job))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    report = {
        "schema_version": 2,
        "job": job,
        "run": run,
        "group": group,
        "status": "running",
        "current_step": "dense_observations",
        "policy": policy.record(),
        "generation_policy": generation_policy.__dict__,
        "inputs": store.reference_artifact(store.resolve(inputs["manifest"])),
        "assets": assets,
        "artifacts": {},
        "limitations": [
            "Image landmarks and facial depth are network predictions, not independent measurements",
            "Fitted scale and focal length remain uncalibrated; no millimetre accuracy claim",
            "Front-facing normals/alpha are occlusion heuristics; glasses and hidden features remain uncertain",
            "No fitted hair, intrinsic skin albedo, personalized ears/rear head, rig or MetaHuman conversion",
            "Held-out-view reprojection is conditional on fitted camera/expression, not ground-truth 3D likeness",
        ],
    }
    diagnostics = ReconstructionReport(policy, ReconstructionInputs(len(entries)))
    atomic_json(manifest, report)
    try:
        model = HeadModel(
            store.artifact_path(assets["gnm"]["artifact"]),
            store.artifact_path(assets["correspondence"]["artifact"]),
            policy.expression_modes,
        )
        report["model_audit"] = model.audit()
        observations = collect_photo_landmark_observations(
            store,
            entries,
            extraction,
            store.artifact_path(assets["landmarker"]["artifact"]),
        )
        observed_path = root / "observations.json"
        atomic_json(
            observed_path,
            {"method": "mediapipe-478-reversible-source-v1", "faces": observations},
        )
        report["artifacts"]["observations"] = store.reference_artifact(observed_path)
        diagnostics.inputs.landmarks_ready = sum(
            o["status"] == "ready" for o in observations
        )
        preparation = {}
        try:
            views, excluded = prepare_views(
                model,
                observations,
                entries,
                reject_extreme_pose=False,
                report=preparation,
            )
        finally:
            diagnostics.prepared_view_ids = preparation.get("prepared_view_ids", [])
            diagnostics.excluded = [
                ExcludedHeadView(item["face_id"], item["reason"])
                for item in preparation.get("excluded", [])
            ]
        report["observation_counts"] = {
            "inputs": len(entries),
            "dense_ready": sum(o["status"] == "ready" for o in observations),
            "fit_eligible": len(views),
            "excluded": excluded,
        }

        def retain_candidate(modes, candidate):
            """Retain this job's typed candidate without coupling the fitter to artifact storage.
            Keep the established candidate filenames for explicit review.
            """
            path = root / f"candidate-{modes}.npz"
            save_fit(path, candidate)
            report["artifacts"][f"candidate_{modes}"] = store.reference_artifact(path)

        report["current_step"] = "fit_shared_head_identity"
        atomic_json(manifest, report)
        result = fit_shared_head_identity(
            model,
            views,
            policy,
            diagnostics,
            on_candidate=retain_candidate,
        )
        mesh = construct_neutral_head_mesh(model, result, diagnostics)
        vertices = mesh.vertices
        diagnostics.status = "awaiting_human_likeness_review"
        report["fit"] = result.summary.to_dict()
        report["mesh_quality"] = diagnostics.mesh_quality.to_dict()
        save_fit(root / "fit.npz", result)
        export_obj(root / "neutral-head.obj", vertices, model.triangles)
        report["artifacts"].update(
            mesh=store.reference_artifact(root / "neutral-head.obj"),
            parameters=store.reference_artifact(root / "fit.npz"),
        )
        camera_path = root / "cameras.json"
        save_fitted_photo_cameras(camera_path, model, result, views)
        report["artifacts"]["cameras"] = store.reference_artifact(camera_path)
        report["artifacts"].update(
            write_photo_fit_previews(store, root, entries, observations, views, result)
        )
        report["current_step"] = "render_prior_and_fitted_head"
        atomic_json(manifest, report)
        artifacts, render = render_clay_portrait(
            store, root, model, vertices, tools, generation_policy
        )
        report["artifacts"].update(artifacts)
        report["render"] = render
        gallery = load_filter(
            store, extraction["backend"], ProcessingPolicy(**extraction["policy"])
        )
        if (
            gallery is None
            or gallery.fingerprint != inputs["inputs"]["curation_sha256"]
        ):
            raise ValueError(
                "Portrait review changed during fitting; generated result is stale"
            )
        dependencies = [entry["source"] for entry in entries.values()]
        dependencies += [observation["aligned_cutout"] for observation in observations]
        if not all(store.verify(artifact) for artifact in dependencies):
            raise ValueError("Photo/analysis input changed during fitting")
        if not all(store.verify(asset["artifact"]) for asset in assets.values()):
            raise ValueError("Head model assets changed during fitting")
        report.update(
            status="awaiting_human_likeness_review",
            current_step="review_neutral_mesh_and_overlays",
            input_dependencies=dependencies,
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
        RuntimeError,
    ) as error:
        report.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        diagnostics.status = "failed"
        diagnostics.failure = ReconstructionFailure(type(error).__name__, str(error))
    finally:
        report["reconstruction"] = diagnostics.to_dict()
        report["total_seconds"] = round(time.monotonic() - started, 3)
        finalize_job_resources(store, report, manifest)
        atomic_json(manifest, report)
    return report

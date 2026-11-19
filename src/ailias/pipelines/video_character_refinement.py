"""Repeatable local video-to-saved-UE-Character execution and built-in verification."""

from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import time
import uuid
from ailias.common.code_assets import module_source
from ailias.common.execution import StageJournal

import numpy as np

from ailias.metahuman.character_features import (
    apply_character_features,
    character_package,
)
from ailias.metahuman.character_features_worker import validate_profile
from ailias.metahuman.sculpt_response import measure_response
from ailias.head_reconstruction.geometry.video_identity_fitting import (
    fit_native_video,
    surface_quality,
)
from ailias.head_reconstruction.evidence.video_sampling import extract_evidence
from ailias.generation.unreal import readiness
from ailias.storage import atomic_json


MAXIMUM_MESH_CM = 0.75
MAXIMUM_PREDICTION_ERROR_CM = 0.02
RUN_SUCCESS = "native_video_completed_awaiting_likeness_review"
STAGES = (
    "preflight",
    "evidence",
    "calibration",
    "fit",
    "apply_reload",
    "mesh_export",
    "verification",
)


def verify_mesh(prediction_path, exported_root):
    """Evaluate the actual native mesh, never substitute predicted geometry for it."""
    actual = np.fromfile(exported_root / "reference.f32", "<f4").reshape(-1, 3)
    triangles = np.asarray(
        json.loads((exported_root / "triangles.json").read_text())
    ).reshape(-1, 3)
    with np.load(prediction_path, allow_pickle=False) as predicted:
        if (
            actual.shape != predicted["reference"].shape
            or not np.isfinite(actual).all()
            or not np.array_equal(triangles, predicted["triangles"])
        ):
            raise ValueError(
                "Actual native geometry dimensions, finiteness or topology differ"
            )
        movement = np.linalg.norm(actual - predicted["reference"], axis=1)
        error = np.linalg.norm(actual - predicted["predicted"], axis=1)
        quality = surface_quality(predicted["reference"], actual, triangles)
    result = {
        "vertex_count": len(actual),
        "topology_unchanged": True,
        "surface": quality,
        "maximum_displacement_cm": float(movement.max()),
        "prediction_error_cm_quantiles": np.quantile(error, [0.5, 0.95, 1]).tolist(),
    }
    if (
        not quality["passed"]
        or movement.max() > MAXIMUM_MESH_CM
        or error.max() > MAXIMUM_PREDICTION_ERROR_CM
    ):
        raise ValueError(
            "Actual native mesh failed surface, displacement or prediction-agreement gates"
        )
    return result


def inspect_video_refinement(store, revision):
    """Recheck retained inputs/outputs without rerunning models or starting Unreal."""
    if not isinstance(revision, str) or not revision.isalnum():
        raise ValueError("Invalid native video run revision")
    path = store.resolve(
        str(Path("facefitting") / "video-runs" / revision / "manifest.json")
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for reference in [
        *report.get("input_dependencies", []),
        *report.get("artifacts", {}).values(),
    ]:
        try:
            if not store.verify(reference):
                problems.append(reference["path"])
        except (OSError, ValueError, KeyError):
            problems.append(reference.get("path", "invalid_artifact_reference"))
    if report["status"] == RUN_SUCCESS and any(
        report.get("stages", {}).get(name, {}).get("status") != "passed"
        for name in STAGES
    ):
        problems.append("incomplete_stage_graph")
    return {**report, "integrity_problems": sorted(set(problems))}


def refine_character_from_video(
    store,
    folder,
    character,
    appearance_path,
    correspondence_path,
    editor,
    project,
    tools,
    video_policy,
    fit_policy,
    *,
    response_path=None,
):
    """Run all machine gates without prompts. Caller owns the session writer lock."""
    root = store.resolve(str(Path("facefitting") / "video-runs" / uuid.uuid4().hex))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "revision": root.name,
        "status": "running",
        "scope": "local video to saved native Character, not assembled runtime animation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "input_dependencies": [],
        "artifacts": {},
        "video_policy": asdict(video_policy),
        "fit_policy": fit_policy.record(),
        "source_character": character,
        "likeness_verified": False,
        "criteria": {
            "minimum_held_out_improvement": 0.01,
            "maximum_category_regression": 0.05,
            "maximum_mesh_displacement_cm": MAXIMUM_MESH_CM,
            "maximum_prediction_error_cm": MAXIMUM_PREDICTION_ERROR_CM,
            "require_fresh_reload": True,
            "require_unchanged_source": True,
            "require_unchanged_topology": True,
            "allowed_flipped_or_collapsed_triangles": 0,
            "self_intersections_checked": False,
        },
        "limitations": [
            "Numerical/asset integrity checks do not prove personal likeness.",
            "Outputs are an editable native Character and exported mesh; runtime assembly, "
            "animation and rendered-image validation are not included.",
            "1 FPS does not retain every brief expression; original video/audio are preserved.",
            "No self-intersection test, measured depth or calibrated camera/skin reflectance.",
        ],
    }
    atomic_json(manifest, report)

    journal = StageJournal(
        report,
        manifest,
        lambda event: print(
            json.dumps({key: event[key] for key in ("stage", "status", "seconds")}),
            file=sys.stderr,
            flush=True,
        ),
    )
    stage = journal.execute

    def retain(name, result, directory):
        path = store.resolve(
            str(Path("facefitting") / directory / result["revision"] / "manifest.json")
        )
        report["artifacts"][name + "_manifest"] = store.reference_artifact(path)
        for key, reference in result.get("artifacts", {}).items():
            report["artifacts"][name + "_" + key] = reference
        report["input_dependencies"].extend(result.get("input_dependencies", []))
        report["stages"][name]["revision"] = result["revision"]
        return path.parent

    try:

        def preflight():
            probe = readiness(editor, project)
            if probe["blockers"]:
                raise ValueError(
                    "Native video prerequisites blocked: "
                    + ", ".join(probe["blockers"])
                )
            if not folder.is_dir():
                raise ValueError("Video input folder does not exist")
            profile = validate_profile(
                json.loads(appearance_path.read_text(encoding="utf-8-sig"))
            )
            report["input_dependencies"] = [
                store.reference_artifact(character_package(project, character)),
                store.reference_artifact(appearance_path),
                store.reference_artifact(correspondence_path),
                store.reference_artifact(project),
                *profile.get("sources", []),
            ]
            if not all(store.verify(item) for item in report["input_dependencies"]):
                raise ValueError("Native video input dependencies changed")
            for name, source_path in (
                (
                    "native_video_pipeline.py",
                    module_source("ailias.pipelines.video_character_refinement"),
                ),
                (
                    "video.py",
                    module_source("ailias.head_reconstruction.evidence.video_sampling"),
                ),
                (
                    "native_video_fit.py",
                    module_source(
                        "ailias.head_reconstruction.geometry.video_identity_fitting"
                    ),
                ),
                (
                    "native_photo_fit.py",
                    module_source("ailias.head_reconstruction.geometry.sculpt_fitting"),
                ),
                (
                    "native_response.py",
                    module_source("ailias.metahuman.sculpt_response"),
                ),
                (
                    "native_response_worker.py",
                    module_source("ailias.metahuman.sculpt_response_worker"),
                ),
                (
                    "native_features.py",
                    module_source("ailias.metahuman.character_features"),
                ),
                (
                    "native_features_worker.py",
                    module_source("ailias.metahuman.character_features_worker"),
                ),
                (
                    "native_correspondence.py",
                    module_source(
                        "ailias.head_reconstruction.geometry.surface_correspondence"
                    ),
                ),
                (
                    "model.py",
                    module_source("ailias.head_reconstruction.geometry.head_model"),
                ),
                (
                    "fitting.py",
                    module_source("ailias.head_reconstruction.geometry.optimization"),
                ),
                ("pipeline.py", module_source("ailias.head_reconstruction.pipeline")),
                ("settings.py", module_source("ailias.head_reconstruction.settings")),
            ):
                report["artifacts"]["code_" + name] = store.copy_artifact(
                    source_path, str(root.relative_to(store.root) / "code")
                )
            report["environment"] = {"python": platform.python_version()}
            for name in ("av", "mediapipe", "numpy", "scipy", "torch"):
                try:
                    report["environment"][name] = version(name)
                except PackageNotFoundError:
                    report["environment"][name] = "not_installed"

        stage("preflight", preflight)

        def evidence():
            result = extract_evidence(store, folder, video_policy)
            retain("evidence", result, "video")
            if result["status"] != "video_evidence_ready":
                raise ValueError("Video evidence lacks required fitting coverage")
            report["input_dependencies"].extend(
                s["source"] for s in result["sources"].values()
            )
            report["evidence_summary"] = {
                "images": len(result["frames"]),
                "dense_detections": sum("source_xy" in f for f in result["frames"]),
                "source_counts": {
                    s["name"]: s["decoded_images"] for s in result["sources"].values()
                },
                "training_views": len(result["fitting_selection"]["training"]),
                "validation_views": len(result["fitting_selection"]["validation"]),
            }
            return result

        stage("evidence", evidence)

        def calibration():
            if response_path is None:
                result = measure_response(store, character, editor, project, tools)
            else:
                result = json.loads(response_path.read_text(encoding="utf-8-sig"))
                report["input_dependencies"].append(
                    store.reference_artifact(response_path)
                )
            retain("calibration", result, "responses")
            report["stages"]["calibration"]["reused"] = response_path is not None
            if (
                result["status"] != "native_response_measured_awaiting_geometry_fit"
                or result["source_character"] != character
                or not result.get("input_dependencies")
                or result["input_dependencies"][0] != report["input_dependencies"][0]
                or not all(store.verify(item) for item in result["input_dependencies"])
            ):
                raise ValueError(
                    "Native response is incomplete, stale or not matched to the requested source"
                )
            return result

        stage("calibration", calibration)

        def fitting():
            with redirect_stdout(sys.stderr):
                result = fit_native_video(
                    store,
                    store.artifact_path(report["artifacts"]["evidence_manifest"]),
                    store.artifact_path(report["artifacts"]["calibration_manifest"]),
                    correspondence_path,
                    appearance_path,
                    fit_policy,
                    maximum_mesh_cm=MAXIMUM_MESH_CM,
                )
            retain("fit", result, "video-fits")
            if result["status"] != "native_video_proposal_ready":
                raise ValueError(
                    "Native video proposal failed its held-out or predicted-mesh gates"
                )
            report["held_out_improvement"] = result["improvement"]
            return result

        stage("fit", fitting)

        def applying():
            result = apply_character_features(
                store,
                character,
                store.artifact_path(report["artifacts"]["fit_native-profile.json"]),
                editor,
                project,
                tools,
                assemble=False,
                cloud_services=False,
            )
            retain("apply_reload", result, "features")
            if (
                result["status"] != "features_reloaded_awaiting_visual_review"
                or not result.get("fresh_editor_reload_verified")
                or result.get("source_character_modified") is not False
            ):
                raise ValueError(
                    "Native save/reload or source preservation did not pass"
                )
            report["character"] = result["character"]
            return result

        applied = stage("apply_reload", applying)

        def exporting():
            result = measure_response(
                store, applied["character"], editor, project, tools, indices=[32]
            )
            retain("mesh_export", result, "responses")
            if result[
                "status"
            ] != "native_response_measured_awaiting_geometry_fit" or not result[
                "native_result"
            ].get("source_unchanged"):
                raise ValueError(
                    "Actual native mesh export or source restoration failed"
                )
            return result

        stage("mesh_export", exporting)

        def verification():
            report["actual_mesh"] = verify_mesh(
                store.artifact_path(report["artifacts"]["fit_prediction.npz"]),
                store.artifact_path(report["artifacts"]["mesh_export_manifest"]).parent,
            )
            checks = [*report["input_dependencies"], *report["artifacts"].values()]
            if not all(store.verify(item) for item in checks):
                raise ValueError(
                    "A retained source, intermediate or final output changed during execution"
                )
            report["source_unchanged"] = True
            report["fresh_editor_reload_verified"] = True

        stage("verification", verification)
        report["status"] = RUN_SUCCESS
    except Exception as error:
        report.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
    finally:
        report["seconds"] = round(time.monotonic() - started, 6)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest, report)
    return report

"""Fit one native identity with per-frame expression/camera nuisance parameters."""

import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np

from ailias.head_reconstruction.geometry.optimization import evaluation_score, fit, prepare_views
from ailias.head_reconstruction.geometry.head_model import HeadModel, model_assets
from ailias.head_reconstruction.geometry.surface_correspondence import rebind_native_faces, sample_native
from ailias.head_reconstruction.geometry.sculpt_fitting import proposed_vertices
from ailias.head_reconstruction.geometry.artifact_io import export_obj, save_fit
from ailias.storage import atomic_json


def align_expression_basis(source, target, expressions, supported):
    """Similarity alignment changes expression coordinates, not native identity."""
    source, target, expressions = map(lambda value: np.asarray(value, dtype=float),
                                     (source, target, expressions))
    supported = np.asarray(supported, dtype=bool)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3
            or expressions.ndim != 3 or expressions.shape[1:] != source.shape
            or supported.shape != (len(source),) or supported.sum() < 100
            or not all(np.isfinite(v).all() for v in (source, target, expressions))):
        raise ValueError("Expression alignment needs finite matched supported face points")
    a, b = source[supported]-source[supported].mean(0), target[supported]-target[supported].mean(0)
    u, singular, vt = np.linalg.svd(a.T @ b)
    correction = np.ones(3)
    correction[-1] = np.linalg.det(u @ vt)
    rotation = (u*correction) @ vt
    scale = float((singular*correction).sum()/max(float((a*a).sum()), 1e-12))
    if not .25 < scale < 4:
        raise ValueError("Native/GNM expression alignment has an unsupported scale")
    return expressions @ rotation*scale, {"rotation": rotation.tolist(), "scale": scale,
                                         "calibrated": False}


def geometry_score(result, views):
    """Category balance prevents many frontal neighbours from hiding side regressions."""
    return evaluation_score(result["report"], views)["per_initial_view_category"]


def surface_quality(reference, candidate, triangles):
    """Reject collapsed/flipped predicted faces; this does not test self-intersections."""
    before, after = (np.asarray(value)[triangles] for value in (reference, candidate))
    normals = [np.cross(value[:, 1]-value[:, 0], value[:, 2]-value[:, 0])
               for value in (before, after)]
    areas = [np.linalg.norm(value, axis=1) for value in normals]
    active = areas[0] > 1e-8
    ratio = areas[1][active]/areas[0][active]
    if not len(ratio) or not np.isfinite(ratio).all():
        raise ValueError("Native prediction has no valid comparable surface")
    flipped = int(np.sum(np.einsum("ij,ij->i", *normals)[active] <= 0))
    quantiles = np.quantile(ratio, [.01, .5, .99])
    return {"flipped_or_collapsed_triangles": flipped, "area_ratio_quantiles": quantiles.tolist(),
            "passed": flipped == 0 and bool(quantiles[0] > .1 and quantiles[-1] < 10),
            "self_intersections_checked": False}


def fit_native_video(store, evidence_path, response_path, correspondence_path, appearance_path, policy,
                     *, maximum_mesh_cm=.75):
    """Create an unapplied native sculpt profile, never a likeness approval or a Blender scene."""
    root = store.resolve(str(Path("facefitting") / "video-fits" / uuid.uuid4().hex))
    root.mkdir(parents=True, exist_ok=False)
    report = {"revision": root.name, "status": "running", "artifacts": {},
              "native_edits_applied": False, "likeness_verified": False, "policy": policy.record()}
    manifest_path = root / "manifest.json"
    atomic_json(manifest_path, report)
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
        response = json.loads(response_path.read_text(encoding="utf-8-sig"))
        if response["status"] != "native_response_measured_awaiting_geometry_fit":
            raise ValueError("Native geometry calibration has not completed")
        assets = model_assets(store)
        dependencies = [
            store.reference_artifact(evidence_path), store.reference_artifact(response_path),
            store.reference_artifact(correspondence_path), store.reference_artifact(appearance_path),
            *evidence["artifacts"].values(), *[s["source"] for s in evidence["sources"].values()],
            *response["artifacts"].values(), *response["input_dependencies"],
            *[asset["artifact"] for asset in assets.values()],
        ]
        if not all(store.verify(item) for item in dependencies):
            raise ValueError("Video, appearance, model or native response inputs changed")
        report["input_dependencies"] = dependencies
        response_root = response_path.parent
        reference = np.fromfile(response_root / "reference.f32", "<f4").reshape(-1, 3).astype(float)
        native_result = response["native_result"]
        response_count = len(native_result["responses"])
        responses = np.memmap(response_root / "responses.f32", "<f4", "r",
                              shape=(response_count, len(reference), 3))
        triangles = np.asarray(json.loads((response_root / "triangles.json").read_text())).reshape(-1, 3)
        with np.load(correspondence_path, allow_pickle=False) as archive:
            cached = dict(archive)
        correspondence = rebind_native_faces(cached.pop("triangles"), triangles, cached)
        supported = correspondence["supported"].copy()
        native_points = np.zeros((468, 3))
        native_points[supported], _ = sample_native(reference, triangles, correspondence)
        native_points[~supported] = native_points[supported].mean(0)
        faces = triangles[correspondence["triangle_indices"][supported]]
        point_responses = np.zeros((response_count, 468, 3))
        point_responses[:, supported] = np.einsum(
            "kpvc,pv->kpc", responses[:, faces], correspondence["barycentric"][supported])

        def generic(values):
            return np.asarray(values)[..., [0, 2, 1]]*.01

        points, point_responses = generic(native_points), generic(point_responses)
        prior = HeadModel(store.artifact_path(assets["gnm"]["artifact"]),
                          store.artifact_path(assets["correspondence"]["artifact"]), policy.expression_modes)
        supported &= prior.supported
        expressions, alignment = align_expression_basis(
            prior.points, points, prior.expression_points, supported)
        normals = prior.normals @ np.asarray(alignment["rotation"])
        u, singular, _ = np.linalg.svd(point_responses[:, supported].reshape(response_count, -1),
                                      full_matrices=False)
        max_modes = max(policy.identity_candidates)
        if max_modes > len(singular) or singular[max_modes-1] < singular[0]*1e-6:
            raise ValueError("Native response has insufficient independent modes")
        control_modes = u[:, :max_modes].T
        identity_points = np.einsum("mk,kpc->mpc", control_modes, point_responses)
        model = SimpleNamespace(points=points, center=points[supported].mean(0),
                                identity_points=identity_points, expression_points=expressions,
                                normals=normals, supported=supported,
                                regions=prior.regions, region_names=prior.region_names)
        selected = evidence["fitting_selection"]
        training_ids, validation_ids = set(selected["training"]), set(selected["validation"])
        selected_frames = [f for f in evidence["frames"] if f["face_id"] in training_ids | validation_ids]
        observations = [{**frame, "status": "ready"} for frame in selected_frames]
        entries = {f["face_id"]: {} for f in observations}
        views, excluded = prepare_views(model, observations, entries)
        training = [v for v in views if v["face_id"] in training_ids]
        validation = [v for v in views if v["face_id"] in validation_ids]
        if len(training) < 3 or not validation or not any(
                v["view"]["category"] == "front" for v in validation):
            raise ValueError("Need training coverage and held-out frontal evidence")
        report.update(
            source_character=response["source_character"], expression_alignment=alignment,
            excluded=excluded, training=[v["face_id"] for v in training],
            validation=[v["face_id"] for v in validation],
            method="native-response shared identity + aligned GNM per-frame expressions and camera",
            limitations=[
                "Native render-ray correspondences are approximate semantics, not scanned anatomy.",
                "GNM expression modes are similarity-aligned approximations, not the native facial rig.",
                "Camera focal lengths, distortion, scale and skin reflectance are not calibrated.",
                "Held-out video frames have temporal guards but are not independent capture ground truth.",
                "Full-mesh output is a linear response prediction until actual native save/reload/export.",
                "Reprojection improvement is not visual likeness acceptance.",
            ])
        atomic_json(manifest_path, report)
        print("Fitting source camera/expression reference", flush=True)
        baseline_training = fit(model, training, policy, 0, fixed_identity=np.zeros(0))
        baseline_validation = fit(model, validation, policy, 0, fixed_identity=np.zeros(0))
        baseline_score = geometry_score(baseline_validation, validation)
        report["baseline_validation"] = baseline_validation["report"]
        best = None
        report["candidates"] = []
        for modes in policy.identity_candidates:
            print(f"Fitting {modes} native identity modes; expressions remain per-frame", flush=True)
            initial = np.r_[np.zeros(modes), baseline_training["per_view"].ravel()]
            candidate = fit(model, training, policy, modes, initial=initial)
            control_values = candidate["identity"] @ control_modes[:modes]
            predicted, clipped = proposed_vertices(reference, responses, control_values, maximum_mesh_cm)
            mesh_quality = surface_quality(reference, predicted, triangles)
            for _ in range(8):
                if mesh_quality["passed"]:
                    break
                clipped *= .5
                predicted = reference+(predicted-reference)*.5
                mesh_quality = surface_quality(reference, predicted, triangles)
            scale = min(1., np.linalg.norm(clipped)/max(np.linalg.norm(control_values), 1e-12))
            identity = candidate["identity"]*scale
            result = fit(model, validation, policy, modes, fixed_identity=identity,
                         initial=baseline_validation["per_view"].ravel())
            score = geometry_score(result, validation)
            improvement = 1-float(np.mean(list(score.values())))/float(np.mean(list(baseline_score.values())))
            nonregression = all(score.get(category, float("inf")) <= value*1.05
                                for category, value in baseline_score.items())
            record = {"identity_modes": modes, "improvement": improvement,
                      "category_nonregression": nonregression, "mesh_budget_scale": scale,
                      "predicted_surface_quality": mesh_quality,
                      "training": candidate["report"], "validation": result["report"]}
            report["candidates"].append(record)
            print(json.dumps({k:record[k] for k in ("identity_modes", "improvement", "category_nonregression")}),
                  flush=True)
            atomic_json(manifest_path, report)
            if (mesh_quality["passed"] and nonregression and improvement >= .01
                    and (best is None or improvement > best[0])):
                best = improvement, identity, clipped, predicted, candidate, result
        if best is None:
            report["status"] = "geometry_quality_gate_failed"
            return report
        improvement, identity, controls, predicted, candidate, validation_result = best
        profile = json.loads(appearance_path.read_text(encoding="utf-8-sig"))
        grouped = controls.reshape(-1, 3)
        maximum_control = float(np.linalg.norm(grouped, axis=1).max())
        limit = max(maximum_mesh_cm, maximum_control)*1.1
        if limit > 2:
            raise ValueError("Native proposal exceeds the supported control safety budget")
        profile["sculpt"] = {
            "reference_landmarks": native_result["reference_landmarks"],
            "deltas": [{"index": index, "delta_cm": grouped[position].tolist()}
                       for position, index in enumerate(native_result["indices"])],
            "max_displacement_cm": limit,
        }
        # Pin the complete evidence chain, including external originals, in the native revision.
        profile["sources"] = dependencies
        atomic_json(root / "native-profile.json", profile)
        export_obj(root / "predicted-native-head.obj", predicted, triangles,
                   description="predicted native head; native axes, centimetres; not a measured personal scan")
        np.savez_compressed(root / "prediction.npz", reference=reference, predicted=predicted,
                            controls_cm=grouped, triangles=triangles)
        save_fit(root / "training-fit.npz", candidate)
        save_fit(root / "validation-fit.npz", validation_result)
        for name in ("native-profile.json", "predicted-native-head.obj", "prediction.npz",
                     "training-fit.npz", "validation-fit.npz"):
            report["artifacts"][name] = store.reference_artifact(root / name)
        report.update(status="native_video_proposal_ready", improvement=improvement,
                      selected_identity_modes=len(identity), maximum_control_cm=maximum_control,
                      maximum_predicted_mesh_cm=float(np.linalg.norm(predicted-reference, axis=1).max()))
        if not all(store.verify(item) for item in dependencies):
            raise ValueError("Input evidence changed during fitting")
        return report
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        atomic_json(manifest_path, report)

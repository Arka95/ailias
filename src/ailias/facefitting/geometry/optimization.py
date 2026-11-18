"""Prepare cameras and visibility heuristics, fit heads and score held-out views.

Owns the SciPy CPU reference and explicit dispatch to the CUDA fitting backend.
"""

import math
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def project(points, cameras, center):
    matrices = Rotation.from_rotvec(cameras[:, :3]).as_matrix()
    camera_points = np.einsum("nvi,nji->nvj", points - center, matrices) + cameras[:, None, 3:6]
    depth = camera_points[:, :, 2]
    return (np.exp(cameras[:, None, 6:7]) * camera_points[:, :, :2]
            / np.maximum(depth[:, :, None], 1e-5)), depth


def dense_view(camera, center, points):
    rotation = Rotation.from_rotvec(camera[:3]).as_matrix()
    direction = -rotation.T @ camera[3:6] + center - points.mean(0)
    yaw = math.degrees(math.atan2(direction[0], direction[2]))
    pitch = math.degrees(math.atan2(direction[1], math.hypot(direction[0], direction[2])))
    if abs(pitch) > 20 and abs(yaw) < 35:
        category = "up" if pitch > 0 else "down"
    elif abs(yaw) <= 20:
        category = "front"
    else:
        category = ("left_" if yaw < 0 else "right_") + ("oblique" if abs(yaw) < 55 else "profile")
    return {"yaw_deg": round(yaw, 2), "pitch_deg": round(pitch, 2), "category": category,
            "calibrated": False, "method": "dense-gnm-perspective-v1",
            "laterality": "camera on canonical -X/+X; not anatomical side certification"}


def initialize_camera(points, target, center, supported, eye_span):
    import cv2

    best = None
    indices = np.flatnonzero(supported)
    if len(indices) < 100:
        raise ValueError("Insufficient supported dense landmarks")
    objects = np.ascontiguousarray(points[indices] - center)
    image = np.ascontiguousarray(target[indices])
    for focal in (0.7, 1.2, 1.8):
        intrinsic = np.diag([focal, focal, 1.0])
        success, rvec, tvec = cv2.solvePnP(objects, image, intrinsic, None,
                                         flags=cv2.SOLVEPNP_SQPNP)
        if not success:
            continue
        camera = np.r_[rvec.ravel(), tvec.ravel(), math.log(focal)]
        prediction, depth = project(points[None], camera[None], center)
        error = float(np.median(np.linalg.norm(prediction[0, indices] - image, axis=1)) / eye_span)
        if np.all(depth > 0) and (best is None or error < best[0]):
            best = error, camera
    if best is None:
        raise ValueError("Could not initialize a positive-depth dense head camera")
    return best[1]


def prepare_views(model, observations, entries):
    prepared, excluded = [], []
    for observation in observations:
        if observation["status"] != "ready":
            excluded.append({"face_id": observation["face_id"], "reason": observation["reason"]})
            continue
        entry = entries[observation["face_id"]]
        size = np.asarray(observation["source_size"])
        scale = max(size)
        target = (np.asarray(observation["source_xy"])[:468] - size / 2) / scale
        eye_span = observation["eye_span_source"] / scale
        supported = model.supported & np.asarray(observation["on_head"])[:468]
        try:
            camera = initialize_camera(model.points, target, model.center, supported, eye_span)
        except ValueError as error:
            excluded.append({"face_id": observation["face_id"], "reason": str(error)})
            continue
        center = -Rotation.from_rotvec(camera[:3]).as_matrix().T @ camera[3:6] + model.center
        facing = np.sum(model.normals * (center - model.points), axis=1)
        facing /= np.maximum(np.linalg.norm(center - model.points, axis=1), 1e-12)
        # A front-facing prior normal is only a self-occlusion heuristic, not visibility ground truth.
        supported &= facing > 0.1
        if int(supported.sum()) < 100:
            excluded.append({"face_id": observation["face_id"], "reason": "insufficient_front_facing_support"})
            continue
        counts = np.bincount(model.regions[supported], minlength=len(model.region_names))
        balance = np.clip(supported.sum() / np.maximum(counts[model.regions], 1) / (counts > 0).sum(), .25, 4)
        weights = supported * balance
        weights /= weights.sum()
        lighting = entry.get("lighting", {}).get("baseline_penalty", 0)
        view = dense_view(camera, model.center, model.points)
        if abs(view["yaw_deg"]) > 95 or abs(view["pitch_deg"]) > 55:
            excluded.append({"face_id": observation["face_id"], "reason": "extreme_or_ambiguous_dense_pose"})
            continue
        prepared.append({
            "face_id": observation["face_id"], "camera": camera, "target": target,
            "eye_span": eye_span, "weights": weights, "source_size": size,
            "view_weight": (1 if view["category"] == "front" else .85) / (1 + lighting),
            "view": view,
        })
    if len(prepared) < 5:
        raise ValueError("Need at least five reliable dense views for shared-identity fitting")
    return prepared, excluded


def validation_indices(views):
    groups = {}
    for i, view in enumerate(views):
        groups.setdefault(view["view"]["category"], []).append(i)
    heldout = []
    for group in groups.values():
        ordered = sorted(group, key=lambda i: views[i]["face_id"])
        heldout.extend(ordered[4::5])
    if not heldout:
        heldout = [len(views) - 1]
    return sorted(heldout)


def evaluation_score(report, views):
    categories = {v["face_id"]: v["view"]["category"] for v in views}
    errors = {}
    for metric in report["views"]:
        errors.setdefault(categories[metric["face_id"]], []).append(metric["median_outer_eye_span_error"])
    means = {category: float(np.mean(values)) for category, values in errors.items()}
    return {"view_balanced_mean_error": float(np.mean(list(means.values()))),
            "per_initial_view_category": means,
            "median_view_error": report["median_outer_eye_span_error"]}


def fit(model, views, policy, identity_modes, fixed_identity=None, initial=None):
    if policy.device != "cpu":
        from ailias.facefitting.geometry.cuda_optimization import fit_cuda
        return fit_cuda(model, views, policy, identity_modes, fixed_identity, initial)
    started = time.monotonic()
    count, point_count, expression_count = len(views), len(model.points), len(model.expression_points)
    variable_identity = identity_modes if fixed_identity is None else 0
    block = 7 + expression_count
    points = model.points.copy()
    if fixed_identity is not None:
        points += np.einsum("i,ivc->vc", fixed_identity, model.identity_points[:len(fixed_identity)])
    target = np.asarray([v["target"] for v in views])
    eye_span = np.asarray([v["eye_span"] for v in views])[:, None, None]
    view_weights = np.asarray([v["view_weight"] for v in views], dtype=float)
    view_weights /= view_weights.sum()
    weights = np.asarray([v["weights"] for v in views]) * view_weights[:, None]
    root_weights = np.sqrt(weights[:, :, None])
    cameras = np.asarray([v["camera"] for v in views])
    x = np.r_[np.zeros(variable_identity), np.c_[cameras, np.zeros((count, expression_count))].ravel()]
    if initial is not None:
        x = initial.copy()
    lower = np.r_[np.full(variable_identity, -2.5),
                  np.tile(np.r_[[-7]*3, [-5, -5, .06], math.log(.45), [-3]*expression_count], count)]
    upper = np.r_[np.full(variable_identity, 2.5),
                  np.tile(np.r_[[7]*3, [5, 5, 10], math.log(3), [3]*expression_count], count)]
    x = np.clip(x, lower + 1e-8, upper - 1e-8)

    def unpack(parameters):
        identity = parameters[:variable_identity]
        per_view = parameters[variable_identity:].reshape(count, block)
        world = points[None] + np.einsum("ni,ivc->nvc", per_view[:, 7:], model.expression_points)
        if variable_identity:
            world += np.einsum("i,ivc->vc", identity, model.identity_points[:variable_identity])
        return identity, per_view, world

    def residual(parameters):
        identity, per_view, world = unpack(parameters)
        projected, depth = project(world, per_view[:, :7], model.center)
        error = (projected - target) / eye_span
        magnitude = np.linalg.norm(error, axis=2, keepdims=True)
        robust = np.sqrt(np.minimum(1, .05 / np.maximum(magnitude, 1e-12)))
        identity_prior = (identity * np.sqrt(policy.identity_regularization)
                          * (1 + np.arange(variable_identity) / 40))
        expression_prior = (per_view[:, 7:] * np.sqrt(
            policy.expression_regularization * view_weights[:, None]))
        depth_penalty = np.maximum(.015 - depth, 0) * root_weights[:, :, 0] * 10
        return np.r_[(error * robust * root_weights).ravel(), identity_prior,
                     expression_prior.ravel(), depth_penalty.ravel()]

    observed_rows = count * point_count * 2
    expression_start = observed_rows + variable_identity
    depth_start = expression_start + count * expression_count
    sparsity = lil_matrix((depth_start + count * point_count, len(x)), dtype=int)
    for i in range(count):
        rows = slice(i*point_count*2, (i+1)*point_count*2)
        columns = slice(variable_identity + i*block, variable_identity + (i+1)*block)
        sparsity[rows, :variable_identity] = 1
        sparsity[rows, columns] = 1
        sparsity[expression_start+i*expression_count:expression_start+(i+1)*expression_count,
                 variable_identity+i*block+7:variable_identity+(i+1)*block] = 1
        sparsity[depth_start+i*point_count:depth_start+(i+1)*point_count, :variable_identity] = 1
        sparsity[depth_start+i*point_count:depth_start+(i+1)*point_count, columns] = 1
    for i in range(variable_identity):
        sparsity[observed_rows+i, i] = 1
    optimized = least_squares(
        residual, x, bounds=(lower, upper), jac_sparsity=sparsity.tocsr(), x_scale="jac",
        max_nfev=policy.max_nfev, ftol=2e-5, xtol=2e-5, gtol=1e-7,
        tr_options={"atol": 1e-5, "btol": 1e-5})
    identity, per_view, world = unpack(optimized.x)
    projected, depth = project(world, per_view[:, :7], model.center)
    errors = np.linalg.norm((projected - target) / eye_span, axis=2)
    if (not np.isfinite(optimized.x).all() or np.any(depth <= 0)
            or not np.isfinite(projected).all()):
        raise ValueError("Fitting produced invalid geometry/cameras")
    return summarize(model, views, identity_modes, expression_count,
                     identity if fixed_identity is None else np.asarray(fixed_identity),
                     per_view, projected, world, optimized.x, {
                         "backend": "scipy-cpu", "device": "cpu",
                         "seconds": round(time.monotonic()-started, 3),
                         "nfev": int(optimized.nfev), "converged": bool(optimized.success),
                         "termination": str(optimized.message), "objective": float(optimized.cost),
                     })


def summarize(model, views, identity_modes, expression_count, identity, per_view,
              projected, world, parameters, runtime):
    target = np.asarray([v["target"] for v in views])
    eye_span = np.asarray([v["eye_span"] for v in views])[:, None, None]
    errors = np.linalg.norm((projected - target) / eye_span, axis=2)
    metrics = []
    for i, view in enumerate(views):
        active = view["weights"] > 0
        source_error = np.linalg.norm(projected[i] - target[i], axis=1) * max(view["source_size"])
        metrics.append({
            "face_id": view["face_id"], "supported_points": int(active.sum()),
            "median_outer_eye_span_error": float(np.median(errors[i, active])),
            "p90_outer_eye_span_error": float(np.quantile(errors[i, active], .9)),
            "median_source_pixel_error": float(np.median(source_error[active])),
            "focal_factor": float(np.exp(per_view[i, 6])),
            "view": dense_view(per_view[i, :7], model.center, world[i]),
        })
    return {
        "identity": identity,
        "per_view": per_view, "projected": projected, "parameters": parameters,
        "report": {
            "identity_modes": identity_modes, "expression_modes": expression_count,
            "views": metrics, **runtime,
            "median_outer_eye_span_error": float(np.median([m["median_outer_eye_span_error"] for m in metrics])),
        },
    }

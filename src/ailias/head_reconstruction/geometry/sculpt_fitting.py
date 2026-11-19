"""Fit bounded native response-space corrections, with photo pose kept separate."""

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ailias.head_reconstruction.geometry.optimization import initialize_camera, project
from ailias.imageprocessing.appearance_features import MEASUREMENTS


PAIR_WEIGHTS = {
    "outer_eye_span": 1., "inner_eye_span": 1., "nose_width": 1.,
    "mouth_width": .4, "face_width": .5, "forehead_to_chin": .25,
}


def fit_photo_pose(points, observation, supported, focal_factor):
    """Fit extrinsics only; preserve the separately estimated, uncalibrated focal."""
    points = np.asarray(points, dtype=float)
    size = np.asarray(observation["source_size"], dtype=float)
    target = (np.asarray(observation["source_xy"], dtype=float)[:len(points)]-size/2)/max(size)
    active = np.asarray(supported, dtype=bool) & np.asarray(observation["on_head"], dtype=bool)[:len(points)]
    span = observation["eye_span_source"]/max(size)
    if (points.shape != (468, 3) or active.shape != (468,) or active.sum() < 100
            or not np.isfinite(points).all() or not np.isfinite(target).all()
            or not math.isfinite(focal_factor) or not .1 <= focal_factor <= 10 or span <= 0):
        raise ValueError("Native pose fitting requires supported finite dense points and a positive focal")
    center = points[active].mean(0)
    camera = initialize_camera(points, target, center, active, span)
    log_focal = math.log(focal_factor)

    def residual(extrinsics):
        value = np.r_[extrinsics, log_focal]
        predicted, _ = project(points[None], value[None], center)
        return ((predicted[0, active]-target[active])/span).ravel()

    fit = least_squares(
        residual, camera[:6], loss="soft_l1", f_scale=.025, max_nfev=100,
        bounds=(np.r_[[-7]*3, [-5, -5, .06]], np.r_[[7]*3, [5, 5, 10]]),
    )
    camera = np.r_[fit.x, log_focal]
    predicted, depth = project(points[None], camera[None], center)
    if not fit.success or np.any(depth[0, active] <= 0):
        raise ValueError("Native photo pose did not converge in front of the camera")
    return {
        "face_id": observation["face_id"], "camera": camera, "center": center,
        "target": target, "supported": active, "eye_span": span,
        "focal_calibrated": False,
    }


def corrective_step(points, basis, views, *, ridge=.0001, max_control_cm=.5):
    """Solve a local correction; paired dimensions supplement aggregate pixel error."""
    points, basis = np.asarray(points, dtype=float), np.asarray(basis, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or basis.ndim != 3
            or basis.shape[1:] != points.shape or basis.shape[0] == 0 or basis.shape[0] % 3
            or not np.isfinite(points).all() or not np.isfinite(basis).all()
            or not views or not math.isfinite(ridge) or ridge <= 0
            or not math.isfinite(max_control_cm) or max_control_cm <= 0):
        raise ValueError("Native correction requires finite grouped response modes and positive bounds")
    matrices, targets = [], []
    for view in views:
        camera, center = np.asarray(view["camera"]), np.asarray(view["center"])
        target, active = np.asarray(view["target"]), np.asarray(view["supported"], dtype=bool)
        span = float(view["eye_span"])
        if (target.shape != (len(points), 2) or active.shape != (len(points),)
                or not active.any() or not np.isfinite(target).all() or not math.isfinite(span) or span <= 0):
            raise ValueError("Native correction photo targets or support are invalid")
        rotation = Rotation.from_rotvec(camera[:3]).as_matrix()
        camera_points = (points-center) @ rotation.T+camera[3:6]
        if np.any(camera_points[active, 2] <= 0):
            raise ValueError("Native correction includes points behind a photo camera")
        camera_basis = np.einsum("kpc,dc->kpd", basis, rotation)
        z = np.maximum(camera_points[:, 2], 1e-5)
        focal = math.exp(camera[6])
        predicted = focal*camera_points[:, :2]/z[:, None]
        derivative = focal*(
            camera_basis[:, :, :2]/z[None, :, None]
            - camera_points[None, :, :2]*camera_basis[:, :, 2:3]/z[None, :, None]**2)
        weight = 1/math.sqrt(active.sum()*len(views))
        matrices.append(derivative[:, active].reshape(len(basis), -1).T/span*weight)
        targets.append(((target[active]-predicted[active])/span*weight).ravel())
        if len(points) != 468:
            continue
        for name, pair_weight in PAIR_WEIGHTS.items():
            first, second = MEASUREMENTS[name]
            if not active[[first, second]].all():
                continue
            difference = predicted[first]-predicted[second]
            length = np.linalg.norm(difference)
            if length < 1e-8:
                continue
            desired = np.linalg.norm(target[first]-target[second])
            jacobian = (derivative[:, first]-derivative[:, second]) @ (difference/length)
            pair_scale = math.sqrt(pair_weight/len(views))/span
            matrices.append(jacobian[None]*pair_scale)
            targets.append(np.array([(desired-length)*pair_scale]))
    matrix, target = np.concatenate(matrices), np.concatenate(targets)
    values = np.linalg.lstsq(
        np.r_[matrix, math.sqrt(ridge)*np.eye(len(basis))],
        np.r_[target, np.zeros(len(basis))], rcond=1e-7,
    )[0]
    grouped = values.reshape(-1, 3)
    largest = float(np.linalg.norm(grouped, axis=1).max())
    if largest > max_control_cm:
        values *= max_control_cm/largest
    before = float(np.mean(target**2))
    after = float(np.mean((matrix @ values-target)**2))
    if not np.isfinite(values).all() or after >= before:
        raise ValueError("Native response correction did not improve its local objective")
    return values, {"linear_objective_before": before, "linear_objective_after": after,
                    "maximum_requested_control_cm": float(np.linalg.norm(values.reshape(-1, 3), axis=1).max()),
                    "native_edits_applied": False, "likeness_verified": False}


def proposed_vertices(reference, responses, values, maximum_cm):
    """Bound the whole merged native mesh, not just its selected sculpt controls."""
    reference, responses, values = np.asarray(reference), np.asarray(responses), np.asarray(values)
    if (reference.ndim != 2 or reference.shape[1] != 3
            or responses.shape != (len(values), *reference.shape)
            or not np.isfinite(reference).all() or not np.isfinite(responses).all()
            or not np.isfinite(values).all() or not math.isfinite(maximum_cm) or maximum_cm <= 0):
        raise ValueError("Invalid native response proposal or whole-mesh budget")
    delta = np.einsum("k,kvc->vc", values, responses)
    maximum = float(np.linalg.norm(delta, axis=1).max())
    scale = min(1., maximum_cm/max(maximum, 1e-12))
    return reference+delta*scale, values*scale

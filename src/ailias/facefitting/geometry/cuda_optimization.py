"""Accelerate shared-identity head fitting with batched CUDA autodiff and Schur/LM.

Keeps camera/expression blocks per view and never silently falls back to CPU.
"""

import math
import time

import numpy as np


def load_torch(device):
    import torch

    index = int(device.split(":")[1])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested {device} unavailable; no CPU fallback")
    return torch


def rotations(torch, vectors):
    x, y, z = vectors.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(-1, 3, 3)
    angle = torch.linalg.vector_norm(vectors, dim=-1)[:, None, None]
    return (torch.eye(3, dtype=vectors.dtype, device=vectors.device)[None]
            + torch.sinc(angle/math.pi) * skew
            + .5 * torch.sinc(angle/(2*math.pi)).square() * (skew @ skew))


def project(torch, points, cameras, center):
    camera = (points-center) @ rotations(torch, cameras[:, :3]).transpose(-2, -1)
    camera = camera + cameras[:, None, 3:6]
    depth = camera[:, :, 2]
    xy = cameras[:, None, 6:7].exp() * camera[:, :, :2] / depth[:, :, None].clamp_min(1e-5)
    return xy, depth


def fit_cuda(model, views, policy, identity_modes, fixed_identity=None, initial=None):
    started = time.monotonic()
    torch = load_torch(policy.device)
    device = torch.device(policy.device)
    count, point_count = len(views), len(model.points)
    expression_count = len(model.expression_points)
    variable_identity = identity_modes if fixed_identity is None else 0
    block = 7 + expression_count

    def tensor(value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)

    points = tensor(model.points)
    identity_basis = tensor(model.identity_points[:max(variable_identity, len(fixed_identity)
                                                     if fixed_identity is not None else 0)].reshape(-1, point_count*3))
    expressions = tensor(model.expression_points.reshape(expression_count, -1))
    center = tensor(model.center)
    if fixed_identity is not None and len(fixed_identity):
        points = points + (tensor(fixed_identity) @ identity_basis).reshape(point_count, 3)
    target = tensor([v["target"] for v in views])
    eye_span = tensor([v["eye_span"] for v in views])[:, None, None]
    view_weights = tensor([v["view_weight"] for v in views])
    view_weights = view_weights / view_weights.sum()
    weights = tensor([v["weights"] for v in views]) * view_weights[:, None]
    root_weights = weights.sqrt()[:, :, None]
    identity_prior = tensor(np.sqrt(policy.identity_regularization)
                            * (1 + np.arange(variable_identity)/40))
    cameras = np.asarray([v["camera"] for v in views])
    initial_values = np.r_[np.zeros(variable_identity),
                           np.c_[cameras, np.zeros((count, expression_count))].ravel()]
    if initial is not None:
        initial_values = initial
    lower = tensor(np.r_[np.full(variable_identity, -2.5),
                         np.tile(np.r_[[-7]*3, [-5, -5, .06], math.log(.45),
                                       [-3]*expression_count], count)])
    upper = tensor(np.r_[np.full(variable_identity, 2.5),
                         np.tile(np.r_[[7]*3, [5, 5, 10], math.log(3),
                                       [3]*expression_count], count)])
    fraction = ((tensor(initial_values)-lower)/(upper-lower)).clamp(1e-5, 1-1e-5)
    raw = torch.logit(fraction)
    shared = raw[:variable_identity]
    local = raw[variable_identity:].reshape(count, block)
    local_lower = lower[variable_identity:variable_identity+block]
    local_upper = upper[variable_identity:variable_identity+block]
    id_lower, id_upper = lower[:variable_identity], upper[:variable_identity]
    evaluations = 0

    def unpack(shared_raw, local_raw):
        identity = id_lower + (id_upper-id_lower)*shared_raw.sigmoid()
        per_view = local_lower + (local_upper-local_lower)*local_raw.sigmoid()
        parameters = torch.cat((identity, per_view.reshape(-1)))
        world = points[None] + (per_view[:, 7:] @ expressions).reshape(count, point_count, 3)
        if variable_identity:
            world = world + (identity @ identity_basis).reshape(point_count, 3)
        return parameters, identity, per_view, world

    def view_residual(shared_raw, local_raw, observed, span, weight, view_weight):
        identity = id_lower + (id_upper-id_lower)*shared_raw.sigmoid()
        parameters = local_lower + (local_upper-local_lower)*local_raw.sigmoid()
        world = points + (parameters[7:] @ expressions).reshape(point_count, 3)
        if variable_identity:
            world = world + (identity @ identity_basis).reshape(point_count, 3)
        predicted, depth = project(torch, world[None], parameters[None, :7], center)
        error = (predicted[0]-observed)/span
        magnitude = torch.linalg.vector_norm(error, dim=1, keepdim=True).clamp_min(1e-12)
        robust = (.05/magnitude).clamp_max(1).sqrt()
        return torch.cat(((error*robust*weight).reshape(-1),
                          identity*identity_prior*view_weight.sqrt(),
                          parameters[7:]*(policy.expression_regularization*view_weight).sqrt(),
                          ((.015-depth[0]).clamp_min(0)*weight[:, 0]*10)))

    def residual_and_aux(*args):
        residual = view_residual(*args)
        return residual, residual

    dimensions = (None, 0, 0, 0, 0, 0)
    residual_batch = torch.func.vmap(view_residual, in_dims=dimensions)
    differentiate = torch.func.jacfwd(
        residual_and_aux, argnums=(0, 1) if variable_identity else 1, has_aux=True)
    jacobian_batch = torch.func.vmap(differentiate, in_dims=dimensions)
    span = eye_span[:, 0, 0]
    damping, converged, termination = .01, False, "iteration limit"
    jacobians = 0
    for iteration in range(policy.max_iterations):
        jacobian, residual = jacobian_batch(shared, local, target, span, root_weights, view_weights)
        jacobians += 1
        evaluations += 1
        if variable_identity:
            js, jl = jacobian
        else:
            jl = jacobian
            js = jl.new_empty(count, jl.shape[1], 0)
        local_h = jl.transpose(1, 2) @ jl
        local_g = (jl.transpose(1, 2) @ residual[:, :, None])
        shared_h = (js.transpose(1, 2) @ js).sum(0)
        shared_g = (js.transpose(1, 2) @ residual[:, :, None]).sum(0)
        cross = js.transpose(1, 2) @ jl
        cost = float(residual.square().sum()*.5)
        gradient_max = max(float(local_g.abs().max()),
                           float(shared_g.abs().max()) if variable_identity else 0)
        if gradient_max < 1e-7:
            converged, termination = True, "gradient tolerance"
            break
        accepted = False
        for attempt in range(8):
            regularized = local_h + torch.diag_embed(
                damping*local_h.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8))
            factor, info = torch.linalg.cholesky_ex(regularized)
            if bool(info.any()):
                damping *= 10
                continue
            local_inverse_g = torch.cholesky_solve(local_g, factor)
            if variable_identity:
                local_inverse_cross = torch.cholesky_solve(cross.transpose(1, 2), factor)
                schur = (shared_h + torch.diag(damping*shared_h.diagonal().clamp_min(1e-8))
                         - (cross @ local_inverse_cross).sum(0))
                schur = .5*(schur+schur.T)
                shared_factor, shared_info = torch.linalg.cholesky_ex(schur)
                if bool(shared_info.any()):
                    damping *= 10
                    continue
                shared_step = -torch.cholesky_solve(
                    shared_g-(cross @ local_inverse_g).sum(0), shared_factor)[:, 0]
                local_step = -local_inverse_g[:, :, 0] - (
                    local_inverse_cross @ shared_step[None, :, None])[:, :, 0]
            else:
                shared_step = shared
                local_step = -local_inverse_g[:, :, 0]
            trial_shared = (shared+shared_step).clamp(-14, 14)
            trial_local = (local+local_step).clamp(-14, 14)
            trial_residual = residual_batch(
                trial_shared, trial_local, target, span, root_weights, view_weights)
            trial_cost = float(trial_residual.square().sum()*.5)
            evaluations += 1
            if math.isfinite(trial_cost) and trial_cost < cost:
                shared, local = trial_shared, trial_local
                damping = max(damping/3, 1e-6)
                accepted = True
                if cost-trial_cost <= 2e-5*max(cost, 1e-12):
                    converged, termination = True, "objective tolerance"
                break
            damping *= 10
        if converged:
            break
        if not accepted:
            termination = "no decreasing damped step; inspect quality gates"
            break
    with torch.no_grad():
        parameters, identity, per_view, world = unpack(shared, local)
        predicted, depth = project(torch, world, per_view[:, :7], center)
        if (not bool(torch.isfinite(parameters).all()) or not bool(torch.isfinite(predicted).all())
                or bool((depth <= 0).any())):
            raise ValueError("CUDA fitting produced invalid geometry/cameras")
    final_residual = residual_batch(shared, local, target, span, root_weights, view_weights)
    objective = float(final_residual.square().sum()*.5)
    torch.cuda.synchronize(device)
    identity_result = identity.detach().cpu().numpy() if fixed_identity is None else np.asarray(fixed_identity)
    from ailias.facefitting.geometry.optimization import summarize
    return summarize(
        model, views, identity_modes, expression_count, identity_result,
        per_view.detach().cpu().numpy(), predicted.detach().cpu().numpy(),
        world.detach().cpu().numpy(), parameters.detach().cpu().numpy(), {
            "backend": "torch-cuda-autodiff-schur-lm", "device": str(device),
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device), "dtype": "float32",
            "batch_views": count, "autograd_device": str(jl.device),
            "nfev": evaluations+1, "iterations": iteration+1, "jacobian_evaluations": jacobians,
            "converged": converged, "termination": termination,
            "objective": objective, "seconds": round(time.monotonic()-started, 3),
            "peak_process_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        })

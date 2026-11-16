"""Project expression-aware photo evidence onto UV texels with CUDA visibility checks.

Produces observed colour and explicit coverage, not lighting-free skin albedo.
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from ailias.facefitting.cuda import load_torch
from ailias.facefitting.model import vertex_normals


def bake(model, identity, uv_map, selected, entries, photos, device):
    if not selected or len(selected) != len(photos):
        raise ValueError("Photo evidence must match the nonempty selected view list")
    torch = load_torch(device)
    import torch.nn.functional as functional

    start = time.monotonic()
    rows, columns, triangle_ids, barycentric = uv_map
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=device)
    indices = tensor(model.triangles[triangle_ids], torch.long)
    bary = tensor(barycentric)
    total = torch.zeros((len(rows), 3), device=device)
    support = torch.zeros(len(rows), device=device)
    contributors = torch.zeros(len(rows), device=device)
    hair_numerator = np.zeros(len(model.vertices))
    hair_denominator = np.zeros(len(model.vertices))
    hair_lengths, hair_colors = [], []
    neutral = model.neutral(identity)
    head_scale = np.linalg.norm(np.ptp(neutral, axis=0))
    anchor = np.asarray(photos[0]["skin_color"])
    records = []
    for view, photo in zip(selected, photos):
        with np.load(photo["data"], allow_pickle=False) as data:
            rgb = data["rgb"].astype(np.float32)/255
            alpha = data["alpha"].astype(np.float32)/255
            labels = data["labels"]
        permitted = np.isin(labels, (1, 2, 3, 4, 5, 7, 8, 10, 12, 13))
        camera = np.asarray(view["camera"])
        world = neutral + np.einsum("i,ivc->vc", view["expression"], model.expression)
        rotation = Rotation.from_rotvec(camera[:3]).as_matrix()
        camera_vertices = (world-model.center) @ rotation.T + camera[3:6]
        normals = vertex_normals(world, model.triangles) @ rotation.T
        points = (tensor(camera_vertices)[indices]*bary[..., None]).sum(1)
        normal = (tensor(normals)[indices]*bary[..., None]).sum(1)
        facing = -(normal*points).sum(1) / (
            normal.norm(dim=1)*points.norm(dim=1)).clamp_min(1e-9)
        source_size = np.asarray(view["source_size"])
        entry = entries[view["face_id"]]
        xy = (points[:, :2]/points[:, 2:].clamp_min(1e-6) * np.exp(camera[6])
              * max(source_size) + tensor(source_size/2))
        xy = (xy-tensor(entry["portrait_bounds"][:2])) * tensor(photo["portrait_scale"])
        height, width = alpha.shape
        rounded = xy.round().long()
        inside = ((rounded[:, 0] >= 0) & (rounded[:, 0] < width)
                  & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
                  & (points[:, 2] > 0))
        pixel = rounded[:, 1].clamp(0, height-1)*width + rounded[:, 0].clamp(0, width-1)
        depth = torch.full((height*width,), float("inf"), device=device)
        depth.scatter_reduce_(0, pixel[inside], points[inside, 2], reduce="amin", include_self=True)
        visible = inside & (points[:, 2] <= depth[pixel]+head_scale*.002)
        grid = ((xy+.5)/tensor([width, height])*2-1)[None, None]
        channels = np.dstack((rgb, alpha*permitted))
        samples = functional.grid_sample(
            tensor(channels).permute(2, 0, 1)[None], grid,
            mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, 0].T
        weight = (visible & (samples[:, 3] >= .98)).float() * facing.clamp_min(0).pow(8)
        weight *= view["appearance_weight"]
        gain = np.clip(anchor/np.maximum(photo["skin_color"], .03), .85, 1.15)
        color = (samples[:, :3]*tensor(gain)).clamp(0, 1)
        linear = torch.where(color <= .04045, color/12.92, ((color+.055)/1.055).pow(2.4))
        total += linear*weight[:, None]
        support += weight
        contributors += weight > .01
        vertex_xy = (camera_vertices[:, :2]/np.maximum(camera_vertices[:, 2:], 1e-6)
                     * np.exp(camera[6])*max(source_size)+source_size/2)
        vertex_xy = (vertex_xy-entry["portrait_bounds"][:2])*photo["portrait_scale"]
        vx, vy = np.rint(vertex_xy).astype(int).T
        vertex_facing = -(normals*camera_vertices).sum(1) / np.maximum(
            np.linalg.norm(normals, axis=1)*np.linalg.norm(camera_vertices, axis=1), 1e-9)
        valid = ((vx >= 0) & (vx < width) & (vy >= 0) & (vy < height)
                 & (vertex_facing > .15) & (camera_vertices[:, 2] > 0))
        vx, vy = np.clip(vx, 0, width-1), np.clip(vy, 0, height-1)
        vertex_depth = depth[tensor(vy*width+vx, torch.long)].cpu().numpy()
        valid &= ((alpha[vy, vx] >= .98) & (permitted[vy, vx] | (labels[vy, vx] == 17))
                  & (camera_vertices[:, 2] <= vertex_depth+head_scale*.002))
        weights = valid*vertex_facing.clip(0)**4*view["appearance_weight"]
        hair_numerator += weights*(labels[vy, vx] == 17)
        hair_denominator += weights
        if photo["hair_color"] is not None:
            hair_colors.append(photo["hair_color"])
            if abs(view["view"]["yaw_deg"]) < 30 and photo["hair_top_pixel"] is not None:
                scalp = (model.groups["skin_exterior"] > .5) & (world[:, 1] > np.quantile(world[:, 1], .9))
                top = np.quantile(vertex_xy[scalp, 1], .05)
                delta = (np.array([0, .01, 0]) @ rotation.T)
                sample = camera_vertices[scalp].mean(0)
                def projected_y(point):
                    return (point[1]/point[2]*np.exp(camera[6])*max(source_size)
                            + source_size[1]/2-entry["portrait_bounds"][1])*photo["portrait_scale"][1]
                rate = abs(projected_y(sample+delta)-projected_y(sample))/.01
                hair_lengths.append(float(np.clip((top-photo["hair_top_pixel"])/max(rate, 1), .008, .06)))
        records.append({"face_id": view["face_id"], "weighted_texels": int((weight > .01).sum()),
                        "relative_color_gain": gain.tolist()})
    covered = support > .01
    linear = total / support[:, None].clamp_min(1e-8)
    color = torch.where(linear <= .0031308, linear*12.92, 1.055*linear.clamp_min(0).pow(1/2.4)-.055)
    torch.cuda.synchronize(device)
    return {
        "color": color.clamp(0, 1).cpu().numpy(), "coverage": covered.cpu().numpy(),
        "weight": support.cpu().numpy(), "contributors": contributors.cpu().numpy(),
        "skin_color": anchor, "hair_probability": hair_numerator/np.maximum(hair_denominator, 1e-8),
        "hair_observed": hair_denominator > 0,
        "hair_color": np.median(hair_colors, axis=0) if hair_colors else np.array([.08, .06, .05]),
        "hair_length": float(np.median(hair_lengths)) if hair_lengths else .015,
        "report": {"backend": "PyTorch CUDA UV projection and dense splat depth test",
                   "device": str(device), "torch": torch.__version__, "seconds": time.monotonic()-start,
                   "gpu": torch.cuda.get_device_name(device),
                   "surface_texels": len(rows), "observed_texels": int(covered.sum()),
                   "observed_fraction": float(covered.float().mean()),
                   "visibility": "facing normals plus UV-sample depth splat, not exact triangle rasterization",
                   "intrinsic_albedo": False, "views": records},
    }

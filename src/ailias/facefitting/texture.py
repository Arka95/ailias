"""Rasterize the head's existing UVs and select photo evidence without changing geometry."""

import math
from pathlib import Path

import numpy as np


def validate_material_textures(path, expected):
    references = [line.partition(" ")[2].strip() for line in path.read_text().splitlines()
                  if line.startswith("map_Kd ")]
    if not references or any((path.parent / Path(name)).resolve() != expected.resolve()
                             for name in references) or not expected.is_file():
        raise ValueError("Exported material does not resolve to the retained skin texture")


def uv_surface_map(uvs, size, supported=None):
    uvs = np.asarray(uvs, dtype=np.float64)
    if (uvs.ndim != 3 or uvs.shape[1:] != (3, 2) or not np.isfinite(uvs).all()
            or np.any(uvs < 0) or np.any(uvs > 1) or type(size) is not int or size < 4):
        raise ValueError("Require finite per-corner UVs in [0,1] and a bounded texture size")
    if size > 4096:
        raise ValueError("Texture size exceeds the bounded raster budget")
    supported = np.ones(len(uvs), dtype=bool) if supported is None else np.asarray(supported, dtype=bool)
    if supported.shape != (len(uvs),):
        raise ValueError("Triangle support must match UV topology")
    triangle = np.full((size, size), -1, dtype=np.int32)
    barycentric = np.zeros((size, size, 3), dtype=np.float32)
    pixels = uvs * [size, -size] + [-.5, size-.5]
    for index in np.flatnonzero(supported):
        a, b, c = pixels[index]
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-10:
            continue
        low = np.maximum(0, np.ceil(pixels[index].min(0)).astype(int))
        high = np.minimum(size-1, np.floor(pixels[index].max(0)).astype(int))
        if np.any(high < low):
            continue
        x, y = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1))
        first = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / denominator
        second = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / denominator
        weights = np.stack((first, second, 1-first-second), axis=-1)
        inside = np.all(weights >= -1e-7, axis=-1)
        ys, xs = y[inside], x[inside]
        triangle[ys, xs] = index
        barycentric[ys, xs] = weights[inside]
    rows, columns = np.nonzero(triangle >= 0)
    if not len(rows):
        raise ValueError("No supported UV surface texels")
    return rows, columns, triangle[rows, columns], barycentric[rows, columns]


def choose_texture_views(cameras, entries, observations, maximum):
    if type(maximum) is not int or not 1 <= maximum <= 64:
        raise ValueError("Texture view count must be in 1..64")
    ranked = []
    for view in cameras:
        entry, observed = entries[view["face_id"]], observations[view["face_id"]]
        scores = observed.get("blendshapes", {})
        expression = sum(scores.get(name, 0) for name in (
            "jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight"))
        lighting = entry.get("lighting", {}).get("baseline_penalty", 0)
        error = view["median_outer_eye_span_error"]
        score = (entry.get("selection_weight", 1) * math.exp(-expression)
                 / ((.01+error) * (1+lighting)))
        ranked.append({**view, "appearance_weight": float(score)})
    ranked.sort(key=lambda view: (-view["appearance_weight"], view["face_id"]))
    chosen, categories = [], set()
    for candidate in ranked:
        category = candidate["view"]["category"]
        if category not in categories and len(chosen) < maximum:
            chosen.append(candidate)
            categories.add(category)
    ids = {view["face_id"] for view in chosen}
    chosen += [view for view in ranked if view["face_id"] not in ids][:maximum-len(chosen)]
    return sorted(chosen, key=lambda view: -view["appearance_weight"])

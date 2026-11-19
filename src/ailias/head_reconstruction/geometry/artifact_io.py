"""Serialize fitted head parameters and UV-preserving OBJ geometry."""

import numpy as np


def export_obj(path, vertices, triangles, *, description="photo-conditioned neutral GNM head; uncalibrated scale",
               triangle_uvs=None):
    if triangle_uvs is not None:
        triangle_uvs = np.asarray(triangle_uvs)
        if triangle_uvs.shape != (len(triangles), 3, 2) or not np.isfinite(triangle_uvs).all():
            raise ValueError("OBJ triangle UVs must match finite per-corner topology")
    with path.open("w", encoding="ascii") as stream:
        stream.write(f"# ailias {description}\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        if triangle_uvs is not None:
            for u, v in triangle_uvs.reshape(-1, 2):
                stream.write(f"vt {u:.9f} {v:.9f}\n")
        for index, (a, b, c) in enumerate(triangles + 1):
            if triangle_uvs is None:
                stream.write(f"f {a} {b} {c}\n")
            else:
                uv = index*3+1
                stream.write(f"f {a}/{uv} {b}/{uv+1} {c}/{uv+2}\n")

def save_fit(path, result):
    np.savez_compressed(path, identity=result["identity"], per_view=result["per_view"],
                        parameters=result["parameters"], projected=result["projected"])

"""Wrap genuine MetaHuman skin topology to approved geometry and rebake photo colour onto its UVs."""

import importlib.util
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform


def main(request):
    root = Path(request["output"])
    root.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("ailias_uv", request["uv_worker"])
    uv_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uv_module)
    source = np.load(request["source_geometry"], allow_pickle=False)
    approved = source["vertices"]
    source_faces = source["triangles"]
    support = np.all(source["skin_exterior"][source_faces] > .5, axis=1)
    supported_faces = source_faces[support]
    source_uvs = source["triangle_uvs"][support]
    surface = BVHTree.FromPolygons(approved.tolist(), supported_faces.tolist(), all_triangles=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=request["native_fbx"])
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(objects) != 1:
        raise ValueError("Require one native LOD0 face mesh with unchanged material sections")
    obj = objects[0]
    mesh = obj.data
    mesh.calc_loop_triangles()
    if "Head" not in mesh.materials[0].name:
        raise ValueError("Native material slot zero is not the verified head skin")
    positions = np.asarray([obj.matrix_world @ vertex.co for vertex in mesh.vertices])
    vertices = positions[:, [0, 2, 1]] * [1, 1, -1]
    faces = np.asarray([list(triangle.vertices) for triangle in mesh.loop_triangles])
    skin = np.asarray([triangle.material_index == 0 for triangle in mesh.loop_triangles])
    faces = faces[skin]
    loops = np.asarray([list(triangle.loops) for triangle in mesh.loop_triangles])[skin]
    uvs = np.asarray([point.uv[:] for point in mesh.uv_layers[0].data])[loops]
    native_ids, inverse = np.unique(faces.ravel(), return_inverse=True)
    head = vertices[native_ids].copy()
    faces = inverse.reshape(-1, 3)
    if len(head) < 24000 or len(head) > 25000:
        raise ValueError("Unexpected native LOD0 skin topology")
    before = head.copy()
    for index, vertex in enumerate(head):
        point, _, _, distance = surface.find_nearest(Vector(vertex))
        if point is None or distance > .04:
            continue
        weight = .95*float(np.clip((vertex[1]-.17)/.035, 0, 1))
        head[index] = vertex*(1-weight) + np.asarray(point)*weight

    def errors(points, target):
        return np.asarray([target.find_nearest(Vector(point))[3] for point in points])

    facial = ((before[:, 1] > .20) & (before[:, 1] < .35) &
              (before[:, 2] > .04) & (np.abs(before[:, 0]) < .085))
    before_error, after_error = errors(before[facial], surface), errors(head[facial], surface)
    # Keep MetaHuman UVs exactly; the native importer uses them to recover DNA vertex order.
    with (root / "wrapped-head.obj").open("w", encoding="ascii") as output:
        for x, y, z in head:
            output.write(f"v {x*100:.9f} {-z*100:.9f} {y*100:.9f}\n")
        for u, v in uvs.reshape(-1, 2):
            output.write(f"vt {u:.9f} {v:.9f}\n")
        for index, face in enumerate(faces):
            corners = [f"{vertex+1}/{index*3+corner+1}" for corner, vertex in enumerate(face)]
            output.write("f " + " ".join(corners) + "\n")
    np.savez(root / "wrapped-head.npz", vertices=head, triangles=faces, triangle_uvs=uvs)

    image = bpy.data.images.load(request["source_texture"], check_existing=False)
    image.colorspace_settings.name = "Non-Color"
    width, height = image.size
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    size = request.get("texture_size", 2048)
    rows, columns, triangle_ids, barycentric = uv_module.uv_surface_map(np.clip(uvs, 0, 1), size)
    queries = np.einsum("ni,nij->nj", barycentric, head[faces[triangle_ids]])
    atlas = np.empty((size, size, 4), dtype=np.float32)
    atlas[:] = [.4, .28, .22, 1]
    coverage = np.zeros((size, size), dtype=bool)
    source_vectors = [tuple(Vector(p) for p in approved[face]) for face in supported_faces]
    uv_vectors = [tuple(Vector((float(u), float(v), 0)) for u, v in face) for face in source_uvs]
    sample_uvs = np.zeros((len(queries), 2), dtype=np.float32)
    valid = np.zeros(len(queries), dtype=bool)
    for index, query in enumerate(queries):
        point, _, triangle, distance = surface.find_nearest(Vector(query))
        if point is None or distance > .04:
            continue
        uv = barycentric_transform(point, *source_vectors[triangle], *uv_vectors[triangle])
        sample_uvs[index] = (uv.x, uv.y)
        valid[index] = True
    selected_uvs = sample_uvs[valid]
    x = np.clip(selected_uvs[:, 0]*width-.5, 0, width-1)
    y = np.clip(selected_uvs[:, 1]*height-.5, 0, height-1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    wx, wy = (x-x0)[:, None], (y-y0)[:, None]
    colours = ((pixels[y0, x0]*(1-wx) + pixels[y0, x1]*wx)*(1-wy) +
               (pixels[y1, x0]*(1-wx) + pixels[y1, x1]*wx)*wy)
    atlas[rows[valid], columns[valid]] = colours
    coverage[rows[valid], columns[valid]] = True
    original_coverage = coverage.copy()
    for _ in range(8):
        total, count = np.zeros_like(atlas), np.zeros_like(coverage, dtype=np.int16)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            shifted = np.roll(coverage, (dy, dx), (0, 1))
            if dy == 1:
                shifted[0] = False
            elif dy == -1:
                shifted[-1] = False
            elif dx == 1:
                shifted[:, 0] = False
            else:
                shifted[:, -1] = False
            total += np.roll(atlas, (dy, dx), (0, 1))*shifted[..., None]
            count += shifted
        fill = ~coverage & (count > 0)
        atlas[fill] = total[fill]/count[fill, None]
        coverage |= fill
    texture = bpy.data.images.new("Photo colour in native MetaHuman UVs", width=size, height=size)
    texture.colorspace_settings.name = "Non-Color"
    texture.pixels.foreach_set(atlas[::-1].copy().ravel())
    texture.filepath_raw = str(root / "metahuman-photo-colour.png")
    texture.file_format = "PNG"
    texture.save()
    report = {
        "status": "wrapped_native_template_awaiting_import", "vertices": len(head),
        "triangles": len(faces), "uvs_preserved": True,
        "facial_vertex_count": int(facial.sum()),
        "before_surface_mm_p50_p95": (np.quantile(before_error, [.5, .95])*1000).tolist(),
        "after_surface_mm_p50_p95": (np.quantile(after_error, [.5, .95])*1000).tolist(),
        "baked_texels": int(original_coverage.sum()), "texture_size": size,
        "photo_colour_is_intrinsic_albedo": False, "likeness_approved": False,
        "measurement": "Nearest approved skin surface, not independent scan or identity accuracy",
    }
    (root / "transfer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[sys.argv.index("--")+1]).read_text()))

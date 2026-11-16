"""Runs inside Blender's bundled Python, not another application virtualenv."""

import json
import math
import sys
from pathlib import Path


def main(request_path):
    import bmesh
    import bpy
    from mathutils import Matrix, Quaternion, Vector

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if bpy.app.version[:2] != (4, 5):
        raise RuntimeError("This worker is verified against Blender 4.5 LTS only")
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=request["mesh"], forward_axis="Y", up_axis="Z")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError("Expected one reconstructed surface, not arbitrary scene content")
    obj = meshes[0]
    obj.name = "HeadCandidate"
    if not all(math.isfinite(c) for vertex in obj.data.vertices for c in vertex.co):
        raise RuntimeError("Mesh contains non-finite vertices")
    before = {"vertices": len(obj.data.vertices), "faces": len(obj.data.polygons)}
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        degenerate = [face for face in bm.faces if face.calc_area() == 0]
        bmesh.ops.delete(bm, geom=degenerate, context="FACES_ONLY")
        loose = [vertex for vertex in bm.verts if not vertex.link_faces]
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        boundary = sum(edge.is_boundary for edge in bm.edges)
        non_manifold = sum(not edge.is_manifold for edge in bm.edges)
        pending = set(bm.verts)
        components = 0
        while pending:
            components += 1
            stack = [pending.pop()]
            while stack:
                vertex = stack.pop()
                for edge in vertex.link_edges:
                    other = edge.other_vert(vertex)
                    if other in pending:
                        pending.remove(other)
                        stack.append(other)
        bm.to_mesh(obj.data)
    finally:
        bm.free()
    obj.data.update()
    if not obj.data.polygons or not obj.data.uv_layers:
        raise RuntimeError("Cleanup requires a real textured surface with retained UVs")
    textures = [
        node.image for material in obj.data.materials if material and material.use_nodes
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image
    ]
    if not textures or any(len(image.pixels) == 0 for image in textures):
        raise RuntimeError("Texture atlas was not loaded; do not export an untextured head")

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    selected_device = "CPU"
    if request["device"] != "CPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = request["device"]
        preferences.refresh_devices()
        devices = [d for d in preferences.devices if d.type == request["device"]]
        index = request["gpu_index"]
        if index >= len(devices):
            raise RuntimeError("Requested Blender GPU is unavailable; CPU fallback is disabled")
        chosen = devices[index]
        for device in preferences.devices:
            device.use = device == chosen
        selected_device = chosen.name
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"
    scene.cycles.samples = request["samples"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.world.color = (0.25, 0.25, 0.25)
    cameras = json.loads(Path(request["cameras"]).read_text(encoding="utf-8"))["cameras"]
    if not cameras:
        raise RuntimeError("A recovered reference camera is required; no invented pose")
    source_id, reference = max(cameras.items(), key=lambda item: item[1]["observation_count"])
    camera = bpy.data.objects.new("RecoveredReference", bpy.data.cameras.new("RecoveredReference"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    # OpenCV camera axes (right, down, forward) to Blender (right, up, backward).
    pose = Matrix(reference["world_to_camera"]).inverted() @ Matrix.Diagonal((1, -1, -1, 1))
    camera.matrix_world = pose
    width, height = reference["portrait_size"]
    scale = request["render_size"] / max(width, height)
    scene.render.resolution_x = max(1, round(width * scale))
    scene.render.resolution_y = max(1, round(height * scale))
    sx, sy = scene.render.resolution_x / width, scene.render.resolution_y / height
    intrinsics = reference["portrait_intrinsics"]
    fx, fy = intrinsics[0][0]*sx, intrinsics[1][1]*sy
    cx, cy = intrinsics[0][2]*sx, intrinsics[1][2]*sy
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36
    camera.data.lens = fx * 36 / scene.render.resolution_x
    aspect = fx / fy
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = aspect
    camera.data.shift_x = (scene.render.resolution_x / 2 - cx) / scene.render.resolution_x
    camera.data.shift_y = (cy - scene.render.resolution_y / 2)*aspect/scene.render.resolution_x

    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    pivot = sum(corners, Vector()) / len(corners)
    diameter = max((corner - pivot).length for corner in corners) * 2
    if diameter <= 0 or not math.isfinite(diameter):
        raise RuntimeError("Mesh has no finite spatial extent")
    light = bpy.data.objects.new("ReviewLight", bpy.data.lights.new("ReviewLight", "AREA"))
    scene.collection.objects.link(light)
    light.location = pose.translation
    light.rotation_euler = (pivot - light.location).to_track_quat("-Z", "Y").to_euler()
    light.data.energy = 200 * diameter**2
    light.data.shape = "DISK"
    light.data.size = 2 * diameter
    camera.data.clip_start = max(diameter/10000, 0.00001)
    camera.data.clip_end = max(diameter*1000, (camera.location - pivot).length*10)
    renders = {}
    up = pose.to_3x3().col[1].normalized()
    for label, degrees in (("reference", 0), ("orbit_45", 45), ("orbit_90", 90)):
        if degrees == 0:
            camera.matrix_world = pose
        else:
            turn = Quaternion(up, math.radians(degrees)).to_matrix().to_4x4()
            camera.matrix_world = Matrix.Translation(pivot) @ turn @ Matrix.Translation(-pivot) @ pose
        bpy.context.view_layer.update()
        scene.render.filepath = str(output / f"{label}.png")
        bpy.ops.render.render(write_still=True)
        renders[label] = {"camera_to_world": [list(row) for row in camera.matrix_world],
                          "human_view_angle_verified": False}
    camera.matrix_world = pose
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=str(output / "head.obj"), export_selected_objects=True,
        forward_axis="Y", up_axis="Z", export_uv=True, export_normals=True,
        export_materials=True, path_mode="RELATIVE",
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "head.blend"))
    result = {
        "status": "complete", "blender_version": bpy.app.version_string,
        "input": before, "vertices": len(obj.data.vertices), "faces": len(obj.data.polygons),
        "degenerate_faces_removed": len(degenerate), "loose_vertices_removed": len(loose),
        "boundary_edges": boundary, "non_manifold_edges": non_manifold,
        "connected_components": components, "smoothing": False, "hole_filling": False,
        "world_units": "unchanged arbitrary SfM units", "metric_scale_verified": False,
        "watertightness_verified": False, "likeness_verified": False,
        "render_backend": "CYCLES", "render_device_requested": request["device"],
        "selected_device": selected_device, "reference_source_id": source_id,
        "render_size": [scene.render.resolution_x, scene.render.resolution_y],
        "renders": renders,
        "limitations": [
            "Floating components and holes require human review; no ears/detail deleted automatically",
            "No head orientation, neutral expression, measured albedo or likeness certification",
            "Only reference uses a recovered camera; orbit views are diagnostic, not reviewed human angles",
        ],
    }
    Path(request["receipt"]).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1])

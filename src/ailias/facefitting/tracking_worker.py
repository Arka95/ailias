"""Render a separate hair-free tracking view without saving or changing a source Blender scene."""

import json
import math
import sys
from pathlib import Path

TRACKING_FOV_DEGREES = 30


def frontal_frame(bounds):
    if not bounds or any(not math.isfinite(value) for point in bounds for value in point):
        raise ValueError("Tracking requires finite, nondegenerate head bounds")
    low = [min(point[axis] for point in bounds) for axis in range(3)]
    high = [max(point[axis] for point in bounds) for axis in range(3)]
    center = [(minimum+maximum)/2 for minimum, maximum in zip(low, high)]
    extent = max(high[0]-low[0], high[1]-low[1])
    if not all(math.isfinite(value) for value in [*low, *high]) or extent <= 0:
        raise ValueError("Tracking requires finite, nondegenerate head bounds")
    distance = extent*1.2/(2*math.tan(math.radians(TRACKING_FOV_DEGREES)/2)) + (high[2]-low[2])/2
    return center, [center[0], center[1], center[2]+distance]


def main(request_path):
    import bpy
    from mathutils import Vector

    request = json.loads(request_path.read_text())
    root = Path(request["output"])
    scene = bpy.context.scene
    head = bpy.data.objects.get("fitted")
    if head is None:
        raise ValueError("Approved fitted head is missing")
    for obj in scene.objects:
        if obj.type in {"MESH", "CURVE", "CURVES"}:
            obj.hide_render = obj != head
    camera = scene.camera
    mode = request.get("camera_mode", "retained")
    if mode == "frontal":
        target, position = frontal_frame([head.matrix_world @ Vector(corner) for corner in head.bound_box])
        camera = camera.copy()
        camera.data = camera.data.copy()
        camera.name = "Ailias frontal tracking camera"
        scene.collection.objects.link(camera)
        scene.camera = camera
        camera.location = Vector(position)
        camera.rotation_euler = (Vector(target)-camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.data.type = "PERSP"
        camera.data.sensor_fit = "HORIZONTAL"
        camera.data.angle = math.radians(TRACKING_FOV_DEGREES)
        camera.data.shift_x = camera.data.shift_y = 0
        camera.data.dof.use_dof = False
    elif mode == "retained":
        if camera.data.type != "ORTHO":
            raise ValueError("Retained tracking expects an orthographic appearance camera; use frontal for a Blender baseline")
        distance = (camera.location-head.matrix_world.translation).length
        width = camera.data.ortho_scale
        camera.data.type = "PERSP"
        camera.data.sensor_fit = "HORIZONTAL"
        camera.data.angle = 2*math.atan(width/(2*distance))
    else:
        raise ValueError("Unknown tracking camera mode")
    scene.render.resolution_x = scene.render.resolution_y = request["size"]
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 16
    if request["device"] == "CPU":
        scene.cycles.device = "CPU"
    else:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = request["device"]
        preferences.get_devices()
        if not any(device.type == request["device"] for device in preferences.devices):
            raise ValueError("Requested tracking-render GPU is unavailable; no CPU fallback")
        for device in preferences.devices:
            device.use = device.type == request["device"]
        scene.cycles.device = "GPU"
    scene.render.filepath = str(root / "tracking-view.png")
    bpy.ops.render.render(write_still=True)
    rotation = camera.matrix_world.to_3x3()
    (root / "tracking-camera.json").write_text(json.dumps({
        "projection": "perspective", "position_gnm": list(camera.matrix_world.translation),
        "forward_gnm": list(-rotation.col[2]), "up_gnm": list(rotation.col[1]),
        "horizontal_fov_degrees": math.degrees(camera.data.angle_x),
        "image_size": [request["size"], request["size"]],
        "render_device": scene.cycles.device,
        "camera_mode": mode,
        "hair_included": False,
    }, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--")+1]))

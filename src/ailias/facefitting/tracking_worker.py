"""Render a hair-free perspective tracking view and its measured camera from an appearance scene."""

import json
import math
import sys
from pathlib import Path

import bpy


def main(request_path):
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
    if camera.data.type != "ORTHO":
        raise ValueError("Expected the retained orthographic review camera")
    distance = (camera.location-head.matrix_world.translation).length
    width = camera.data.ortho_scale
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = 2*math.atan(width/(2*distance))
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
        "hair_included": False,
    }, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--")+1]))

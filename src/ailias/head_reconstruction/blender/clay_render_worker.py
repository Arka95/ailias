"""Owned Blender subprocess: render and save a neutral clay head, not a reconstruction receipt."""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def material(name, color):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = .65
    return result


def main(request):
    config = json.loads(request.read_text(encoding="utf-8"))
    root = Path(config["output"])
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = config["samples"]
    if config["device"] == "CPU":
        scene.cycles.device = "CPU"
    else:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = config["device"]
        preferences.get_devices()
        selected = [d for d in preferences.devices if d.type == config["device"]]
        if not selected:
            raise ValueError("Requested Blender GPU device is unavailable; no CPU fallback")
        for device in preferences.devices:
            device.use = device in selected
        scene.cycles.device = "GPU"
    scene.render.resolution_x = scene.render.resolution_y = config["size"]
    scene.render.resolution_percentage = 100
    scene.world.color = (.18, .18, .18)
    scene.render.image_settings.file_format = "PNG"
    materials = [
        material("Neutral clay - not measured skin albedo", (.42, .32, .28)),
        material("Generic sclera", (.67, .65, .60)),
        material("Generic iris", (.10, .055, .025)),
        material("Generic pupil", (.004, .004, .004)),
        material("Generic teeth", (.70, .66, .55)),
    ]
    with np.load(config["geometry"], allow_pickle=False) as data:
        triangles = data["triangles"]
        objects = {}
        for label in ("prior", "fitted"):
            mesh = bpy.data.meshes.new(label)
            mesh.from_pydata(data[label].tolist(), [], triangles.tolist())
            mesh.update()
            obj = bpy.data.objects.new(label, mesh)
            scene.collection.objects.link(obj)
            for entry in materials:
                mesh.materials.append(entry)
            assignments = np.zeros(len(data[label]), dtype=int)
            for name, index in (("eyes", 1), ("irises", 2), ("pupils", 3), ("teeth", 4)):
                assignments[data[name] > .5] = index
            for polygon, triangle in zip(mesh.polygons, triangles):
                polygon.use_smooth = True
                polygon.material_index = int(np.max(assignments[triangle]))
            objects[label] = obj
    target = Vector((0, .25, .02))
    for name, location, energy, size in (
        ("Key", (-.5, .65, .8), 45, .55), ("Fill", (.6, .35, .4), 22, .5),
        ("Rim", (.2, .65, -.5), 35, .4),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.shape, light.size = energy, "DISK", size
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("Review camera")
    camera = bpy.data.objects.new("Review camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = .40

    def place(angle):
        radians = math.radians(angle)
        camera.location = (math.sin(radians)*.8, .25, math.cos(radians)*.8 + .02)
        forward = (target-camera.location).normalized()
        right = forward.cross(Vector((0, 1, 0))).normalized()
        up = right.cross(forward).normalized()
        camera.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()

    for label, obj in objects.items():
        for name, other in objects.items():
            other.hide_render = other.hide_viewport = name != label
        for angle in (0, 45, 90):
            place(angle)
            scene.render.filepath = str(root / f"{label}-{angle}.png")
            bpy.ops.render.render(write_still=True)
    place(0)
    bpy.context.view_layer.objects.active = objects["fitted"]
    objects["fitted"].select_set(True)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "head-review.blend"))
    (root / "render-runtime.json").write_text(json.dumps({
        "engine": "CYCLES", "requested_device": config["device"],
        "cycles_device": scene.cycles.device, "material": "clay, not measured skin albedo",
        "blender": bpy.app.version_string,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--") + 1]))

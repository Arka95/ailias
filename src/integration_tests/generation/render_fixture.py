"""Blender-only synthetic capture worker for the opt-in native pipeline check."""

import json
import math
import sys
from pathlib import Path


def main(request_path):
    import bpy
    from mathutils import Vector

    request = json.loads(Path(request_path).read_text())
    output = Path(request["output"])
    output.mkdir()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32)
    sphere = bpy.context.object
    sphere.scale = (1, 0.8, 1.2)
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    material = bpy.data.materials.new("SyntheticEmission")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(request["texture"])
    emission = nodes.new("ShaderNodeEmission")
    surface = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(image.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], surface.inputs["Surface"])
    sphere.data.materials.append(material)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.refresh_devices()
    devices = [device for device in preferences.devices if device.type == "CUDA"]
    if not devices:
        raise RuntimeError("CUDA fixture capture unavailable; no CPU fallback")
    for device in preferences.devices:
        device.use = device == devices[0]
    scene.cycles.device = "GPU"
    scene.cycles.samples = 4
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    camera = bpy.data.objects.new("FixtureCamera", bpy.data.cameras.new("FixtureCamera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36
    camera.data.lens = 500 * 36 / 512
    for i, degrees in enumerate(range(-35, 36, 10)):
        angle = math.radians(degrees)
        camera.location = (3.5 * math.sin(angle), -3.5 * math.cos(angle), 0.2 * math.sin(2*angle))
        camera.rotation_euler = (-Vector(camera.location)).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(output / f"capture-{i}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1])

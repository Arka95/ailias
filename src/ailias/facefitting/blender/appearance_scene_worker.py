"""Add UV photo material and evidence-guided approximate hair in a new Blender scene.

The supplied fitted scene and its head geometry are read-only inputs.
"""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def hair_material(color):
    material = bpy.data.materials.new("Photo-sampled hair colour - procedural strand layout")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    linear = np.where(color <= .04045, color/12.92, ((color+.055)/1.055)**2.4)
    shader.inputs["Base Color"].default_value = (*linear, 1)
    shader.inputs["Roughness"].default_value = .55
    return material


def add_hair(scene, vertices, triangles, normals, data, config):
    policy = config["policy"]
    observed = data["hair_observed"].astype(bool)
    probability = data["hair_probability"]
    scalp = data["skin_exterior"] > .5
    eye = float(data["eye_level"])
    supported = scalp & (vertices[:, 1] > eye) & observed & (probability > .5)
    if not supported.any():
        return [], {"status": "no_observed_hair_roots", "strands": 0}
    inferred = (~observed) & (vertices[:, 1] > eye+.025)
    field = np.where(observed, probability, inferred.astype(float))
    edges = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    degree = np.bincount(edges.ravel(), minlength=len(vertices))
    for _ in range(6):
        neighbours = np.zeros(len(vertices))
        np.add.at(neighbours, edges[:, 0], field[edges[:, 1]])
        np.add.at(neighbours, edges[:, 1], field[edges[:, 0]])
        field = .5*field+.5*neighbours/np.maximum(degree, 1)
    roots = scalp & (vertices[:, 1] > eye) & (field > .5)
    faces = triangles[np.all(roots[triangles], axis=1)]
    if len(faces) == 0 or policy["hair_strands"] == 0:
        return [], {"status": "no_supported_hair_roots", "strands": 0}
    length = config["hair_length"]
    cap_vertices = vertices + normals * length*.25
    cap_mesh = bpy.data.meshes.new("Approximate hair volume")
    cap_mesh.from_pydata(cap_vertices.tolist(), [], faces.tolist())
    cap_mesh.update()
    cap = bpy.data.objects.new("Approximate photo-supported hair cap", cap_mesh)
    scene.collection.objects.link(cap)
    material = hair_material(np.asarray(config["hair_color"]))
    cap_mesh.materials.append(material)
    for polygon in cap_mesh.polygons:
        polygon.use_smooth = True
    areas = np.linalg.norm(np.cross(
        vertices[faces[:, 1]]-vertices[faces[:, 0]],
        vertices[faces[:, 2]]-vertices[faces[:, 0]]), axis=1)
    random = np.random.default_rng(policy["seed"])
    ids = random.choice(len(faces), size=policy["hair_strands"], p=areas/areas.sum())
    weights = random.dirichlet([1, 1, 1], size=len(ids))
    selected = faces[ids]
    origins = (cap_vertices[selected]*weights[..., None]).sum(1)
    direction = (normals[selected]*weights[..., None]).sum(1)
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    tangent = np.cross(direction, np.tile([0., 1., 0.], (len(direction), 1)))
    small = np.linalg.norm(tangent, axis=1) < .1
    tangent[small] = np.cross(direction[small], [1., 0., 0.])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    second = np.cross(direction, tangent)
    phases = random.uniform(0, 2*math.pi, (len(origins), 1))
    t = np.linspace(0, 1, policy["hair_segments"])
    strand_length = length*random.uniform(.7, 1.25, (len(origins), 1))
    waves = t[None]*math.pi*2.5+phases
    points = (origins[:, None] + direction[:, None]*strand_length[:, :, None]*t[None, :, None]*.8
              + tangent[:, None]*np.sin(waves)[..., None]*length*.15*t[None, :, None]
              + second[:, None]*np.cos(waves)[..., None]*length*.15*t[None, :, None])
    curves = bpy.data.curves.new("Approximate hair strands", "CURVE")
    curves.dimensions = "3D"
    curves.bevel_depth = .00010
    curves.bevel_resolution = 0
    curves.resolution_u = 1
    for strand in points:
        spline = curves.splines.new("POLY")
        spline.points.add(len(t)-1)
        spline.points.foreach_set("co", np.c_[strand, np.ones(len(t))].astype(np.float32).ravel())
        for index, point in enumerate(spline.points):
            point.radius = 1-.9*index/(len(t)-1)
    hair = bpy.data.objects.new("Hair approximation - not recovered individual strands", curves)
    scene.collection.objects.link(hair)
    curves.materials.append(material)
    return [cap, hair], {"status": "approximation", "strands": len(origins),
                         "supported_root_vertices": int((roots & observed).sum()),
                         "inferred_rear_root_vertices": int((roots & inferred).sum()),
                         "length_model_units": length, "measured_strand_layout": False}


def main(request_path):
    config = json.loads(request_path.read_text())
    root = Path(config["output"])
    scene = bpy.context.scene
    head = bpy.data.objects.get("fitted")
    if head is None or head.type != "MESH":
        raise ValueError("Supplied fitted head object is missing")
    with np.load(config["geometry"], allow_pickle=False) as data:
        vertices = np.asarray([vertex.co[:] for vertex in head.data.vertices])
        deviation = float(np.max(np.abs(vertices-data["vertices"])))
        if deviation > 1e-6:
            raise ValueError("Blender head differs from supplied fitted geometry")
        layer = head.data.uv_layers.new(name="GNM_UV")
        layer.data.foreach_set("uv", data["triangle_uvs"].astype(np.float32).ravel())
        material = bpy.data.materials.new("Observed photo colour - not intrinsic albedo")
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        image = bpy.data.images.load(config["texture"])
        image.pack()
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        uv = material.node_tree.nodes.new("ShaderNodeUVMap")
        uv.uv_map = layer.name
        material.node_tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
        material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        shader.inputs["Roughness"].default_value = .6
        head.data.materials[0] = material
        cornea = bpy.data.materials.new("Generic transparent eye shell - not measured optics")
        cornea.use_nodes = True
        glass = cornea.node_tree.nodes.get("Principled BSDF")
        glass.inputs["Base Color"].default_value = (1, 1, 1, 1)
        glass.inputs["Transmission Weight"].default_value = 1
        glass.inputs["Roughness"].default_value = .03
        glass.inputs["IOR"].default_value = 1.38
        cornea_index = len(head.data.materials)
        head.data.materials.append(cornea)
        for polygon, triangle in zip(head.data.polygons, data["triangles"]):
            if np.all(data["eye_exteriors"][triangle] > .5):
                polygon.material_index = cornea_index
        normals = np.asarray([vertex.normal[:] for vertex in head.data.vertices])
        hair, hair_report = add_hair(scene, vertices, data["triangles"], normals, data, config)
    scene.cycles.samples = config["policy"]["render_samples"]
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = config["device"]
    preferences.get_devices()
    selected = [device for device in preferences.devices if device.type == config["device"]]
    if not selected:
        raise ValueError("Requested appearance-render GPU is unavailable; no CPU fallback")
    for device in preferences.devices:
        device.use = device in selected
    scene.cycles.device = "GPU"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0
    for light in bpy.data.lights:
        light.energy *= .08
    camera = scene.camera
    target = Vector((0, .25, .02))
    scene.render.resolution_x = scene.render.resolution_y = config["size"]
    for angle in (-45, 0, 45, 90):
        radians = math.radians(angle)
        camera.location = (math.sin(radians)*.8, .25, math.cos(radians)*.8+.02)
        forward = (target-camera.location).normalized()
        right = forward.cross(Vector((0, 1, 0))).normalized()
        up = right.cross(forward).normalized()
        camera.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()
        scene.render.filepath = str(root / f"appearance-{angle}.png")
        bpy.ops.render.render(write_still=True)
    camera.location = (0, .25, .82)
    forward = (target-camera.location).normalized()
    camera.rotation_euler = Matrix((Vector((1, 0, 0)), Vector((0, 1, 0)), -forward)).transposed().to_euler()
    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True)
    if hair:
        hair[0].select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.wm.obj_export(filepath=str(root / "textured-head.obj"), export_selected_objects=True,
                          forward_axis="Y", up_axis="Z", export_materials=True, path_mode="STRIP")
    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True)
    original_transform = head.matrix_world.copy()
    try:
        head.matrix_world = Matrix(((0, 0, 1, 0), (1, 0, 0, 0),
                                   (0, 1, 0, 0), (0, 0, 0, 1)))
        bpy.ops.wm.obj_export(filepath=str(root / "head-for-unreal.obj"), export_selected_objects=True,
                              forward_axis="Y", up_axis="Z", global_scale=100,
                              export_materials=True, path_mode="STRIP")
    finally:
        head.matrix_world = original_transform
    groom = "not_exported"
    if hair and bpy.app.build_options.alembic:
        bpy.ops.object.select_all(action="DESELECT")
        hair[-1].select_set(True)
        bpy.ops.wm.alembic_export(filepath=str(root / "hair-groom.abc"), selected=True,
                                  start=1, end=1, as_background_job=False, flatten=True,
                                  curves_as_mesh=False)
        groom = "alembic_curves_exported_not_unreal_validated"
    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "textured-head.blend"))
    (root / "appearance-runtime.json").write_text(json.dumps({
        "blender": bpy.app.version_string, "cycles_device": scene.cycles.device,
        "geometry_maximum_deviation": deviation, "hair": hair_report, "groom": groom,
        "unreal_export": "Hair-free head only; OBJ=(GNM Z,X,Y)*100; UE importer flips Y",
        "unreal_metric_scale_calibrated": False,
        "skin": "photo-derived colour with residual lighting; roughness is a chosen constant",
    }, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--")+1]))

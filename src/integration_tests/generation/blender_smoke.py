"""Opt-in real Blender/GPU check; never imported by the functional unit runner."""

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from ailias.generation import blender_worker
from ailias.generation.settings import load_settings
from ailias.generation.tools import NativeTools
from ailias.storage import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    policy, root = load_settings()
    tools = NativeTools(root, policy.timeout_seconds)
    probe = tools.probe("blender")
    if not probe["ready"]:
        raise ValueError(f"Blender not ready: {probe}")
    vertices = [(1, 0, 3), (-1, 0, 3), (0, 1, 3), (0, -1, 3), (0, 0, 2), (0, 0, 4)]
    faces = [(4, 2, 0), (4, 0, 3), (4, 3, 1), (4, 1, 2),
             (5, 0, 2), (5, 3, 0), (5, 1, 3), (5, 2, 1)]
    with tempfile.TemporaryDirectory(prefix="ailias-blender-check-") as temporary:
        work = Path(temporary)
        mesh = work / "fixture.obj"
        mesh.write_text(
            "mtllib fixture.mtl\nusemtl fixture\n"
            + "".join(f"v {x} {y} {z}\n" for x, y, z in vertices)
            + "vt 0 0\nvt 1 0\nvt 0.5 1\n"
            + "".join("f " + " ".join(f"{v + 1}/{i + 1}" for i, v in enumerate(face)) + "\n"
                      for face in faces), encoding="ascii")
        mesh.with_suffix(".mtl").write_text(
            "newmtl fixture\nKd 1 1 1\nmap_Kd texture.png\n", encoding="ascii")
        texture = Image.new("RGB", (32, 32), "orange")
        draw = ImageDraw.Draw(texture)
        draw.rectangle((0, 0, 15, 15), fill="red")
        draw.rectangle((16, 16, 31, 31), fill="blue")
        texture.save(work / "texture.png")
        camera_path = work / "cameras.json"
        atomic_json(camera_path, {"cameras": {"synthetic-source": {
            "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0],
                                [0, 0, 1, 0], [0, 0, 0, 1]],
            "portrait_size": [256, 256],
            "portrait_intrinsics": [[300, 0, 128], [0, 300, 128], [0, 0, 1]],
            "observation_count": 6,
        }}})
        output = work / "output"
        request = work / "request.json"
        receipt_path = output / "cleanup.json"
        atomic_json(request, {
            "mesh": str(mesh), "output": str(output), "cameras": str(camera_path),
            "receipt": str(receipt_path), "device": policy.blender_device,
            "gpu_index": policy.gpu_index, "render_size": 256, "samples": 4,
        })
        try:
            step = tools.run("blender", "blender-smoke", [
                "--background", "--factory-startup", "--disable-autoexec",
                "--threads", policy.cpu_threads, "--python-exit-code", 1,
                "--python", Path(blender_worker.__file__), "--", request,
            ], work)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            actual_vertices = {
                tuple(map(float, line.split()[1:4])) for line in (output / "head.obj").read_text().splitlines()
                if line.startswith("v ")
            }
            if actual_vertices != set(vertices) or receipt["faces"] != len(faces):
                raise ValueError("Blender changed fixture positions/topology unexpectedly")
            for view in receipt["renders"].values():
                up = [view["camera_to_world"][row][1] for row in range(3)]
                if max(abs(a - b) for a, b in zip(up, (0, -1, 0))) > 0.00001:
                    raise ValueError("Orbit camera changed the recovered reference up direction")
            images = []
            for name in ("reference", "orbit_45", "orbit_90"):
                with Image.open(output / f"{name}.png") as image:
                    rgb = image.convert("RGB")
                    if ImageChops.difference(rgb, Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))).getbbox() is None:
                        raise ValueError(f"{name} produced a flat image, not rendered geometry")
                images.append({"view": name, "sha256": digest(output / f"{name}.png")})
            result = {
                "status": "complete", "fixture": "synthetic textured octahedron; NOT a reconstructed head",
                "tool": probe, "seconds": step["seconds"], "receipt": receipt,
                "unchanged_vertex_positions_verified": True,
                "renders": images, "head_reconstruction_verified": False,
            }
        except Exception as error:
            log = work / "logs" / "blender-smoke.log"
            atomic_json(args.report, {"status": "failed", "error": str(error),
                                      "native_log": log.read_text(errors="replace") if log.exists() else ""})
            raise
    result["temporary_fixture_removed"] = True
    atomic_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

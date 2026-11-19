"""Create a new Blender review scene from the current in-memory fitted geometry."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ailias.storage import atomic_json


def render_clay_portrait(store, root, model, vertices, tools, generation_policy):
    geometry = root / "review-geometry.npz"
    np.savez_compressed(geometry, prior=model.vertices, fitted=vertices, triangles=model.triangles,
                        eyes=model.groups["eyes"], pupils=model.groups["pupils"],
                        irises=model.groups["irises"], teeth=model.groups["teeth"])
    source_worker = Path(__file__).with_name("clay_render_worker.py")
    worker_artifact = store.copy_artifact(source_worker, str(root.relative_to(store.root) / "worker"))
    request = root / "render-request.json"
    atomic_json(request, {"geometry": str(geometry), "output": str(root),
                          "device": generation_policy.blender_device,
                          "size": generation_policy.render_size,
                          "samples": generation_policy.render_samples})
    result = tools.run("blender", "head-review", [
        "--background", "--factory-startup", "--python-exit-code", "1",
        "--python", str(store.artifact_path(worker_artifact)), "--", str(request),
    ], root)
    artifacts = {
        "scene": store.reference_artifact(root / "head-review.blend"),
        "render_request": store.reference_artifact(request),
        "render_worker": worker_artifact,
        "render_geometry": store.reference_artifact(geometry),
        "render_runtime": store.reference_artifact(root / "render-runtime.json"),
    }
    for label in ("prior", "fitted"):
        for angle in (0, 45, 90):
            name = f"{label}-{angle}.png"
            with Image.open(root / name) as image:
                image.verify()
            artifacts[name] = store.reference_artifact(root / name)
    side = generation_policy.render_size
    sheet = Image.new("RGB", (side*3, (side+28)*2), "#181818")
    draw = ImageDraw.Draw(sheet)
    for row, label in enumerate(("prior", "fitted")):
        for column, angle in enumerate((0, 45, 90)):
            with Image.open(root / f"{label}-{angle}.png") as image:
                sheet.paste(image.convert("RGB"), (column*side, row*(side+28)+28))
            draw.text((column*side+8, row*(side+28)+8), f"{label.upper()} / {angle} degrees", fill="white")
    sheet.save(root / "head-comparison.jpg", quality=94)
    artifacts["comparison"] = store.reference_artifact(root / "head-comparison.jpg")
    return artifacts, result

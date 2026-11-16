"""Opt-in real COLMAP CUDA -> dense mesh -> textured OBJ -> Blender GPU check."""

import argparse
import json
import os
import random
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from ailias.generation.pipeline import inspect_job, reconstruct
from ailias.generation.settings import load_settings
from ailias.generation.tools import NativeTools
from ailias.imageprocessing.curation import METHOD, PROFILE, embedding_space
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.storage import Store, atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preview", type=Path, help="Optional retained synthetic reference/render evidence")
    args = parser.parse_args()
    policy, tools_root = load_settings()
    policy = replace(policy, max_images=8, max_feature_image_size=512,
                     max_dense_image_size=400, poisson_depth=8, render_size=256, render_samples=4)
    tools = NativeTools(tools_root, policy.timeout_seconds)
    with tempfile.TemporaryDirectory(prefix="ailias-native-check-") as temporary:
        root = Path(temporary)
        randomizer = random.Random(127)
        noise = Image.new("RGB", (128, 64))
        noise.putdata([tuple(randomizer.randrange(25, 235) for _ in range(3)) for _ in range(128*64)])
        noise.resize((1024, 512), Image.Resampling.BICUBIC).save(root / "texture.png")
        request = root / "capture.json"
        atomic_json(request, {"texture": str(root / "texture.png"), "output": str(root / "photos")})
        tools.run("blender", "fixture-capture", [
            "--background", "--factory-startup", "--disable-autoexec", "--threads", policy.cpu_threads,
            "--python-exit-code", 1, "--python", Path(__file__).with_name("render_fixture.py"),
            "--", request,
        ], root)
        store = Store(root / "data", "synthetic-object")
        backend = {"provider": "synthetic-fixture-not-a-face-model", "revision": "1",
                   "models": {"sface": "not-executed"}}
        run = "synthetic-native-fixture"
        scan = {"schema_version": 1, "status": "complete", "run": run,
                "backend": backend, "policy": asdict(ProcessingPolicy()),
                "sources": {}, "faces": {}, "groups": []}
        examples = []
        for source in sorted((root / "photos").glob("*.png")):
            checksum = digest(source)
            relative = str(Path("imageprocessing") / "assets" / "portraits" / f"{checksum}.png")
            portrait = store.resolve(relative)
            portrait.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, portrait)
            face_id = checksum + "-00"
            quality = {"native_short_side": 320, "laplacian_variance": 100, "reasons": []}
            scan["sources"][checksum] = {"artifact": store.reference_artifact(source)}
            scan["faces"][face_id] = {
                "id": face_id, "source_id": checksum, "quality": quality,
                "products": {"portrait": relative}, "portrait_bounds": [0, 0, 512, 512],
                "box_xywh": [96, 48, 320, 416], "embedding": [1, 0],
                "eligible": True, "identity_ambiguous": False,
            }
            examples.append({
                "portrait_sha256": checksum, "label": "accept", "source_id": checksum,
                "portrait_bounds": [0, 0, 512, 512], "embedding": [1, 0], "quality": quality,
                "artifact": {"path": relative, "sha256": checksum},
            })
        scan["groups"] = [{"id": "fixture-object", "members": list(scan["faces"])}]
        atomic_json(store.resolve(str(Path("imageprocessing") / "runs" / run / "manifest.json")), scan)
        atomic_json(store.resolve(PROFILE), {
            "schema_version": 1, "method": METHOD, "alias": "synthetic-object",
            "embedding_space": embedding_space(backend), "examples": examples,
            "labels": "Known synthetic fixture geometry, NOT human portrait review or face inference",
        })
        with store.lock():
            result = reconstruct(store, run, "fixture-object", tools, policy)
        validation = inspect_job(store, result["job"])
        report = {"fixture": "synthetic textured sphere; NOT a reconstructed human head",
                  "result": result, "integrity_problems": validation["integrity_problems"],
                  "head_reconstruction_verified": False}
        nonflat, panels = {}, []
        if result["status"] == "complete" and not validation["integrity_problems"]:
            source_id = result["blender"]["reference_source_id"]
            panels.append(("Synthetic source", store.artifact_path(scan["sources"][source_id]["artifact"])))
            for label in ("reference", "orbit_45", "orbit_90"):
                image_path = store.artifact_path(result["artifacts"][f"render_{label}"])
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    nonflat[label] = ImageChops.difference(
                        rgb, Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))).getbbox() is not None
                panels.append((label, image_path))
        report["nonflat_render_checks"] = nonflat
        if (result["status"] != "complete" or validation["integrity_problems"]
                or not nonflat or not all(nonflat.values()) or result["cleanup"]["errors"]):
            report["native_logs"] = {
                path.name: path.read_text(errors="replace")[-12000:]
                for path in (Path(result["manifest"]).parent / "logs").glob("*.log")}
            atomic_json(args.report, report)
            raise RuntimeError(f"Native fixture failed; diagnostic report: {args.report}")
        if args.preview:
            sheet = Image.new("RGB", (256*len(panels), 280), "white")
            draw = ImageDraw.Draw(sheet)
            for index, (label, image_path) in enumerate(panels):
                with Image.open(image_path) as image:
                    sheet.paste(image.convert("RGB").resize((256, 256)), (256*index, 24))
                draw.text((256*index + 4, 4), label, fill="black")
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(args.preview)
            report["synthetic_preview"] = str(args.preview)
    report["temporary_fixture_removed"] = True
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

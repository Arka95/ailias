import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from ailias.generation.cameras import read_model
from ailias.generation.inputs import prepare_inputs
from ailias.generation.matching import matching_summary
from ailias.generation.mesh import ply_header, textured_ply_to_obj
from ailias.generation.settings import GenerationPolicy
from ailias.imageprocessing.pipeline import component, new_id
from ailias.resources import cleanup_owned_files
from ailias.storage import Store, atomic_json


class ReconstructionBlocked(ValueError):
    pass


def load_job(store: Store, job: str):
    path = store.resolve(str(Path("generation") / "runs" / component(job) / "manifest.json"))
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("job") != job:
        raise ValueError("Malformed generation manifest")
    return report, path


def inspect_job(store: Store, job: str):
    report, path = load_job(store, job)
    problems = [name for name, artifact in report["artifacts"].items() if not store.verify(artifact)]
    if not store.verify(report["input_view"]):
        problems.append("input_view")
    else:
        inputs = json.loads(store.artifact_path(report["input_view"]).read_text(encoding="utf-8"))
        problems.extend(artifact["path"] for artifact in inputs["files"] if not store.verify(artifact))
        for entry in inputs["inputs"]["entries"]:
            if not store.verify(entry["source"]):
                problems.append(f"source:{entry['source_id']}")
    if not store.verify(report["curation"]):
        problems.append("curation_changed_since_job")
    return {**report, "manifest": str(path), "integrity_problems": problems}


def finalize_job(store: Store, report: dict, path: Path):
    retained = [str(path.relative_to(store.root))]
    retained.extend(artifact["path"] for artifact in report["artifacts"].values()
                    if artifact.get("storage") != "external")
    retained.extend(str(p.relative_to(store.root)) for p in (path.parent / "logs").glob("*.log"))
    if report["status"] == "running":
        report["status"] = "interrupted"
    report["cleanup"] = cleanup_owned_files(
        store, str(path.parent.relative_to(store.root)), retained)
    atomic_json(path, report)


def clean_mesh(store: Store, report: dict, path: Path, tools, policy: GenerationPolicy):
    from ailias.generation import blender_worker

    mesh = report["artifacts"].get("reconstructed_mesh")
    cameras = report["artifacts"].get("cameras")
    if mesh is None or cameras is None:
        raise ValueError("A reconstructed mesh and recovered cameras are required for Blender")
    for name in ("reconstructed_mesh", "reconstructed_material", "texture", "cameras"):
        if not store.verify(report["artifacts"][name]):
            raise ValueError(f"{name} changed since reconstruction")
    output = path.parent / "blender"
    if any(name.startswith("cleaned_") for name in report["artifacts"]):
        raise ValueError("Cleanup already exists; preserve this revision and start another job")
    output.mkdir(exist_ok=True)
    request_path = path.parent / "blender-request.json"
    receipt = output / "cleanup.json"
    atomic_json(request_path, {
        "mesh": str(store.artifact_path(mesh)), "output": str(output),
        "cameras": str(store.artifact_path(cameras)), "receipt": str(receipt),
        "device": policy.blender_device, "gpu_index": policy.gpu_index,
        "render_size": policy.render_size, "samples": policy.render_samples,
    })
    report["current_step"] = "blender-cleanup"
    atomic_json(path, report)
    step = tools.run("blender", "blender-cleanup", [
        "--background", "--factory-startup", "--disable-autoexec",
        "--threads", policy.cpu_threads, "--python-exit-code", 1,
        "--python", Path(blender_worker.__file__), "--", request_path,
    ], path.parent)
    report["steps"].append(step)
    result = json.loads(receipt.read_text(encoding="utf-8"))
    if (result.get("status") != "complete" or result.get("faces", 0) < policy.min_mesh_faces
            or result.get("metric_scale_verified") is not False):
        raise ValueError("Blender did not produce a verified cleanup receipt")
    expected = {"cleaned_mesh": output / "head.obj",
                "cleaned_material": output / "head.mtl",
                "cleaned_scene": output / "head.blend", "cleaned_receipt": receipt}
    artifacts = {name: store.reference_artifact(artifact) for name, artifact in expected.items()}
    for name in ("reference", "orbit_45", "orbit_90"):
        artifact = output / f"{name}.png"
        with Image.open(artifact) as image:
            image.verify()
        artifacts[f"render_{name}"] = store.reference_artifact(artifact)
    report["artifacts"].update(artifacts)
    report["blender"] = result
    return result


def reconstruct(store: Store, run: str, group: str, tools, policy: GenerationPolicy) -> dict:
    readiness = {name: tools.probe(name) for name in ("colmap", "blender")}
    if not all(probe["ready"] for probe in readiness.values()):
        raise ValueError(f"Native tool readiness failed: {readiness}")
    inputs = prepare_inputs(store, run, group, policy)
    job = new_id()
    root = store.resolve(str(Path("generation") / "runs" / job))
    root.mkdir(parents=True, exist_ok=False)
    path = root / "manifest.json"
    report = {
        "schema_version": 1, "job": job, "status": "running",
        "mode": "experimental_candidate_group; not the manual capture workflow",
        "run": run, "group": group,
        "input_view": store.reference_artifact(store.resolve(inputs["manifest"])),
        "curation": {"path": str(Path("imageprocessing") / "curation.json"),
                     "sha256": inputs["inputs"]["curation_sha256"]},
        "policy": asdict(policy), "tools": readiness, "steps": [], "artifacts": {},
        "identity_confirmed": False, "capture_conditions_reviewed": False,
        "metahuman": {"status": "not_started"},
        "production_ready": False, "likeness_verified": False,
    }
    atomic_json(path, report)
    start = time.monotonic()
    work = root / "work"
    work.mkdir()
    output = root / "reconstruction"
    output.mkdir()
    view = store.resolve(inputs["manifest"]).parent
    images, masks = view / "images", view / "masks"
    database, sparse = work / "database.db", work / "sparse"
    sparse.mkdir()

    def colmap(operation, command, options):
        report["current_step"] = operation
        atomic_json(path, report)
        args = [command]
        for key, value in options.items():
            args.extend([f"--{key}", str(value)])
        report["steps"].append(tools.run("colmap", operation, args, root))

    try:
        colmap("feature-extraction", "feature_extractor", {
            "database_path": database, "image_path": images,
            "ImageReader.mask_path": masks, "ImageReader.camera_model": "SIMPLE_PINHOLE",
            "ImageReader.single_camera_per_image": 1, "FeatureExtraction.type": "SIFT",
            "FeatureExtraction.use_gpu": 1, "FeatureExtraction.gpu_index": policy.gpu_index,
            "FeatureExtraction.num_threads": policy.cpu_threads,
            "FeatureExtraction.max_image_size": policy.max_feature_image_size,
        })
        colmap("feature-matching", "exhaustive_matcher", {
            "database_path": database, "FeatureMatching.type": "SIFT_BRUTEFORCE",
            "FeatureMatching.use_gpu": 1, "FeatureMatching.gpu_index": policy.gpu_index,
            "FeatureMatching.num_threads": policy.cpu_threads,
            "FeatureMatching.guided_matching": 1,
        })
        report["matching"] = matching_summary(database)
        report["compute_evidence"] = {
            "GPU_feature_extractor_created": "Creating SIFT GPU feature extractor" in (
                root / "logs" / "feature-extraction.log").read_text(encoding="utf-8"),
            "GPU_feature_matcher_created": "Creating SIFT GPU feature matcher" in (
                root / "logs" / "feature-matching.log").read_text(encoding="utf-8"),
        }
        if not all(report["compute_evidence"].values()):
            raise ValueError("Native logs did not confirm both requested GPU feature operations")
        report["current_step"] = "verified-view-graph-gate"
        count = report["matching"]["largest_connected_component"]
        if (count < policy.min_registered_images
                or count / len(inputs["inputs"]["entries"]) < policy.min_registered_fraction):
            raise ReconstructionBlocked(
                f"Only {count} input images form a connected verified-match component; "
                "insufficient multi-view overlap for the configured reconstruction gate.")
        colmap("sparse-mapping", "mapper", {
            "database_path": database, "image_path": images, "output_path": sparse,
            "Mapper.num_threads": policy.cpu_threads,
            "Mapper.min_model_size": policy.min_registered_images,
        })
        models = []
        for model in sorted(sparse.iterdir()):
            if not model.is_dir() or not (model / "images.bin").is_file():
                continue
            text = work / f"sparse-text-{model.name}"
            text.mkdir()
            colmap(f"export-cameras-{model.name}", "model_converter", {
                "input_path": model, "output_path": text, "output_type": "TXT",
            })
            recovered = read_model(text, inputs)
            models.append((recovered["registered_images"], recovered["sparse_points"],
                           model, text, recovered))
        if not models:
            raise ReconstructionBlocked(
                "COLMAP could not register a stable model. Casual portraits may lack rigid "
                "multi-view overlap; no placeholder head or invented cameras were generated.")
        _, _, model, text, recovered = max(models, key=lambda value: value[:2])
        retained_model = output / "sparse"
        retained_model.mkdir()
        for source in text.glob("*.txt"):
            target = retained_model / source.name
            source.replace(target)
            report["artifacts"][f"sparse_{source.stem}"] = store.reference_artifact(target)
        camera_path = output / "cameras.json"
        atomic_json(camera_path, recovered)
        report["artifacts"]["cameras"] = store.reference_artifact(camera_path)
        report["reconstruction"] = {key: value for key, value in recovered.items() if key != "cameras"}
        if (recovered["registered_images"] < policy.min_registered_images
                or recovered["registered_images"] / recovered["input_images"] < policy.min_registered_fraction
                or recovered["sparse_points"] < policy.min_sparse_points
                or recovered["mean_reprojection_error_px"] is None
                or recovered["mean_reprojection_error_px"] > policy.max_reprojection_error_px):
            raise ReconstructionBlocked(
                "Recovered geometry failed the configured registration/point/reprojection gate; "
                "inspect the saved camera report. No dense head or MetaHuman was claimed.")
        dense = work / "dense"
        colmap("undistortion", "image_undistorter", {
            "image_path": images, "input_path": model, "output_path": dense,
            "output_type": "COLMAP", "max_image_size": policy.max_dense_image_size,
            "num_threads": policy.cpu_threads, "num_patch_match_src_images": 10,
            "copy_policy": "HARD_LINK",
        })
        colmap("dense-stereo", "patch_match_stereo", {
            "workspace_path": dense, "workspace_format": "COLMAP",
            "PatchMatchStereo.gpu_index": policy.gpu_index,
            "PatchMatchStereo.max_image_size": policy.max_dense_image_size,
            "PatchMatchStereo.geom_consistency": 1, "PatchMatchStereo.cache_size": 2,
            "PatchMatchStereo.num_threads": policy.cpu_threads,
        })
        fused = work / "fused.ply"
        colmap("surface-fusion", "stereo_fusion", {
            "workspace_path": dense, "workspace_format": "COLMAP", "input_type": "geometric",
            "output_path": fused, "StereoFusion.mask_path": masks,
            "StereoFusion.num_threads": policy.cpu_threads, "StereoFusion.cache_size": 2,
        })
        count = ply_header(fused)["elements"].get("vertex", 0)
        report["reconstruction"]["fused_points"] = count
        if count < policy.min_fused_points:
            raise ReconstructionBlocked("Too few consistent foreground depth samples; no surface fabricated")
        surface = work / "surface.ply"
        colmap("surface-meshing", "poisson_mesher", {
            "input_path": fused, "output_path": surface,
            "PoissonMeshing.depth": policy.poisson_depth,
            "PoissonMeshing.trim": policy.poisson_trim,
            "PoissonMeshing.num_threads": policy.cpu_threads,
        })
        report["reconstruction"]["meshed_faces"] = ply_header(surface)["elements"].get("face", 0)
        if report["reconstruction"]["meshed_faces"] < policy.min_mesh_faces:
            raise ReconstructionBlocked("Meshed surface is too small to treat as a head candidate")
        textured = work / "textured"
        colmap("texture-atlas", "mesh_texturer", {
            "workspace_path": dense, "input_path": surface, "output_path": textured,
            "output_type": "TXT", "MeshTextureMapping.num_threads": policy.cpu_threads,
        })
        mesh = output / "head.obj"
        report["reconstruction"].update(textured_ply_to_obj(textured / "mesh.ply", mesh))
        texture = output / "texture.png"
        with Image.open(textured / "texture.png") as image:
            image.verify()
        shutil.move(str(textured / "texture.png"), texture)
        for name, artifact in (("reconstructed_mesh", mesh),
                               ("reconstructed_material", mesh.with_suffix(".mtl")),
                               ("texture", texture)):
            report["artifacts"][name] = store.reference_artifact(artifact)
        clean_mesh(store, report, path, tools, policy)
        report["status"] = "complete"
        report["current_step"] = "awaiting_human_mesh_and_likeness_review"
    except ReconstructionBlocked as error:
        report.update(status="blocked", reason=str(error))
    except (OSError, ValueError, TypeError, KeyError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        report["seconds"] = round(time.monotonic() - start, 3)
        finalize_job(store, report, path)
    return {**report, "manifest": str(path)}

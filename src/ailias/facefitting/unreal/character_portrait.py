"""Render a saved feature revision without narration, fitting or animated expression."""

import json
import uuid
from pathlib import Path

from ailias.facefitting.unreal.character_features import character_package
from ailias.generation.unreal import asset_path
from ailias.imageprocessing.pipeline import component
from ailias.reanimation.media import inspect_frames
from ailias.reanimation.pipeline import doctor
from ailias.resources import finalize_resources
from ailias.storage import atomic_json


def render_character_portrait(store, feature_job, editor, project, policy, tools, *, diagnostic=False):
    source_path = store.resolve(str(Path("facefitting") / "features" /
                                    component(feature_job) / "manifest.json"))
    source = json.loads(source_path.read_text())
    blueprint = source.get("blueprint")
    if diagnostic:
        blueprint = (blueprint or source.get("build", {}).get("runtime", {}).get("blueprint")
                     or source.get("apply", {}).get("runtime", {}).get("blueprint"))
    if not blueprint or (not diagnostic and (
            source.get("status") != "features_reloaded_awaiting_visual_review"
            or not source.get("fresh_editor_reload_verified"))):
        raise ValueError("Native review requires a successfully reloaded feature/runtime revision")
    dependencies = [store.reference_artifact(source_path), *source["artifacts"].values(),
                    *source["input_dependencies"]]
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Native review source dependencies changed")
    if project.resolve() != Path(source["readiness"]["project"]).resolve():
        raise ValueError("Use the project containing the saved feature revision")
    blueprint = asset_path(blueprint)
    revision = uuid.uuid4().hex
    root = store.resolve(str(Path("facefitting") / "native-reviews" / revision))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    probe = doctor(editor, project)
    report = {
        "revision": revision, "feature_job": feature_job,
        "status": "blocked" if probe["blockers"] else "running", "readiness": probe,
        "blueprint": blueprint, "input_dependencies": dependencies, "artifacts": {},
        "policy": policy.to_dict(), "framing": policy.framing(), "placement": policy.placement(),
        "likeness_verified": False, "neutral_review": True,
        "diagnostic": diagnostic, "source_feature_status": source["status"],
        "limitations": ["Neutral expression, not expression-matched to a smiling reference",
                        "A rendered image is evidence for review, not proof of personal likeness"],
    }
    atomic_json(manifest, report)
    request_path, receipt, frames = root / "request.json", root / "native-receipt.json", root / "frames"
    try:
        if probe["blockers"]:
            return report
        frames.mkdir()
        from ailias.reanimation import unreal_worker

        report["artifacts"]["worker"] = store.copy_artifact(
            Path(unreal_worker.__file__), str(root.relative_to(store.root) / "worker"))
        request = {
            "destination": f"/Game/Ailias/NativeReviews/R{revision}", "receipt": str(receipt),
            "blueprint": blueprint, "neutral_review": True,
            "policy": policy.to_dict(), "framing": policy.framing(), "placement": policy.placement(),
            "frames": str(frames), "frame_count": 1,
        }
        atomic_json(request_path, request)
        tools.paths["unreal"] = editor.resolve()
        report["runtime"] = tools.run("unreal", "native-neutral-review", [
            str(project.resolve()), f"-ExecutePythonScript={store.artifact_path(report['artifacts']['worker'])}",
            "-unattended", "-nop4", "-nosplash", "-AllowCommandletRendering",
            "-RenderOffscreen", "-d3d12", "-stdout", "-FullStdOutLogOutput",
            "-ExecCmds=r.DefaultFeature.AutoExposure 0",
        ], root, environment={"AILIAS_REANIMATION_REQUEST": str(request_path)})
        result = json.loads(receipt.read_text())
        report["native"] = result
        if (result.get("status") != "frames_rendered" or not result.get("render_success")
                or not result.get("neutral_review")):
            raise ValueError("Native neutral review did not confirm a completed render")
        image = inspect_frames(frames, policy, 1)[0]
        report["artifacts"]["image"] = store.reference_artifact(image)
        for key in ("sequence", "map", "material"):
            path = asset_path(result[key])
            if not path.startswith(request["destination"] + "/"):
                raise ValueError("Native review output escaped its unique namespace")
            package = character_package(project, path)
            if key == "map":
                package = package.with_suffix(".umap")
            report["artifacts"][key] = store.reference_artifact(package)
        if not all(store.verify(item) for item in dependencies):
            raise ValueError("Native review source changed while rendering")
        report["status"] = "native_neutral_rendered_awaiting_visual_review"
    except (OSError, ValueError, TypeError, KeyError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        for path in (request_path, receipt):
            if path.is_file() and path.stat().st_size:
                report["artifacts"][path.name] = store.reference_artifact(path)
        finalize_resources(store, report, manifest)
        atomic_json(manifest, report)
    return report

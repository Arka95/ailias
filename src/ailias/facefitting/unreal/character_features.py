"""Persist isolated native feature revisions and verify them in a fresh editor process."""

import json
import uuid
from pathlib import Path

from ailias.facefitting.unreal.character_features_worker import validate_profile, verify_profile
from ailias.generation.unreal import asset_path, readiness
from ailias.resources import finalize_resources
from ailias.storage import atomic_json


def character_package(project, character):
    package = asset_path(character).split(".", 1)[0]
    return (project.parent / "Content" / Path(*package.removeprefix("/Game/").split("/"))).with_suffix(
        ".uasset")


def apply_character_features(store, character, profile_path, editor, project, tools, *,
                   baseline=None, assemble=False, cloud_services=False, use_working_baseline=True):
    """Clone an existing Character and apply a hash-pinned profile without changing its source."""
    project = project.resolve()
    source_package = character_package(project, character)
    profile = validate_profile(json.loads(profile_path.read_text(encoding="utf-8-sig")))
    dependencies = [store.reference_artifact(source_package),
                    store.reference_artifact(profile_path), *profile.get("sources", [])]
    baseline_path = baseline or store.resolve("facefitting\\working-baseline.json")
    if baseline is not None or (use_working_baseline and baseline_path.is_file()):
        selected = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
        if selected.get("status") != "owner_selected_working_baseline":
            raise ValueError("Native refinement requires a verified working baseline")
        dependencies.extend([store.reference_artifact(baseline_path), *selected["artifacts"].values()])
    if not all(store.verify(reference) for reference in dependencies):
        raise ValueError("Native feature inputs or working baseline changed")
    probe = readiness(editor, project)
    revision = uuid.uuid4().hex
    root = store.resolve(str(Path("facefitting") / "features" / revision))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    destination = asset_path(f"/Game/Ailias/Features/F{revision}")
    candidate = destination + "/Character.Character"
    report = {
        "revision": revision, "source_character": character, "character": candidate,
        "status": "blocked" if probe["blockers"] else "running", "readiness": probe,
        "input_dependencies": dependencies, "artifacts": {}, "likeness_verified": False,
        "source_character_modified": False, "cloud_requested": False, "rig_requested": False,
        "assembly_requested": assemble, "cloud_services_authorized": cloud_services,
        "limitations": [
            "Saved feature properties and compatible grooms require a separate native visual review",
            "An explicit edit profile is not automatic photo-guided geometry fitting",
            "Iris tint multiplies native colour and is not calibrated photo colour",
            "Texture overrides require native UVs; GNM atlases cannot be assigned directly",
        ],
    }
    atomic_json(manifest, report)
    if probe["blockers"]:
        finalize_resources(store, report, manifest)
        atomic_json(manifest, report)
        return report
    try:
        report["artifacts"]["worker"] = store.copy_artifact(
            (Path(__file__).parents[1] / "unreal" / "character_features_worker.py"), str(root.relative_to(store.root) / "worker"))
        report["artifacts"]["profile"] = store.copy_artifact(
            profile_path, str(root.relative_to(store.root) / "profile"))
        if report["artifacts"]["profile"]["sha256"] != dependencies[1]["sha256"]:
            raise ValueError("Native feature profile changed while preparing the revision")
        tools.paths["unreal"] = Path(probe["editor"])
        expected = None
        operations = [("apply", "features_saved"), ("verify", "features_reloaded")]
        for operation, status in operations:
            request, receipt = root / f"{operation}-request.json", root / f"{operation}-receipt.json"
            atomic_json(request, {
                "operation": operation, "source": character, "character": candidate,
                "destination": destination, "profile": profile,
                "receipt": str(receipt), "expected": expected,
                "cloud_services": cloud_services, "blueprint": report.get("blueprint"),
                "assemble": assemble,
            })
            report[f"{operation}_runtime"] = tools.run("unreal", f"native-features-{operation}", [
                probe["project"], f"-ExecutePythonScript={store.artifact_path(report['artifacts']['worker'])}",
                "-unattended", "-nop4", "-nosplash", "-AllowCommandletRendering",
                "-RenderOffscreen", "-d3d12", "-stdout", "-FullStdOutLogOutput",
            ], root, environment={"AILIAS_FEATURE_REQUEST": str(request)})
            result = json.loads(receipt.read_text())
            report[operation] = result
            report["cloud_requested"] |= bool(result.get("cloud_requested"))
            report["rig_requested"] |= bool(result.get("rig_requested"))
            if result.get("status") != status or result.get("character") != candidate:
                raise ValueError("Native worker did not confirm the requested feature revision")
            if (operation == "verify" and assemble
                    and result.get("runtime", {}).get("blueprint") != report.get("blueprint")):
                raise ValueError("Native worker did not independently verify the assembled actor")
            verify_profile(profile, result["values"], expected)
            if not source_package.is_file() or not all(store.verify(item) for item in dependencies):
                raise ValueError("Native feature inputs changed during editing")
            expected = result["values"]
            if operation == "apply" and assemble:
                report["blueprint"] = asset_path(result["blueprint"])
                report["build"] = {key: result[key] for key in ("blueprint", "runtime", "runtime_assets")}
                report["build"]["status"] = "runtime_saved"
                for index, asset in enumerate(result["runtime_assets"]):
                    report["artifacts"][f"runtime_{index}"] = store.reference_artifact(
                        character_package(project, asset))
            report["artifacts"]["character"] = store.reference_artifact(
                character_package(project, candidate))
        report["status"] = "features_reloaded_awaiting_visual_review"
        report["fresh_editor_reload_verified"] = True
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                      project_may_contain_partial_outputs=True)
    finally:
        report["source_character_modified"] = not store.verify(dependencies[0])
        if report["source_character_modified"]:
            report["status"] = "failed"
            report["error"] = {"type": "SourceIntegrityError", "message": "Source Character changed"}
        for operation in ("apply", "build", "verify"):
            for name in (f"{operation}-request.json", f"{operation}-receipt.json"):
                path = root / name
                if path.is_file() and path.stat().st_size:
                    report["artifacts"][name] = store.reference_artifact(path)
        candidate_package = character_package(project, candidate)
        if candidate_package.is_file():
            report["artifacts"]["character"] = store.reference_artifact(candidate_package)
        retained = {item["path"] for item in report["artifacts"].values()}
        for path in candidate_package.parent.rglob("*.uasset"):
            reference = store.reference_artifact(path)
            if reference["path"] not in retained:
                report["artifacts"][f"native_package_{len(retained)}"] = reference
                retained.add(reference["path"])
        finalize_resources(store, report, manifest)
        atomic_json(manifest, report)
    return report

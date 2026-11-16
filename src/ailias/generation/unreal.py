import json
import re
import shutil
from pathlib import Path

from ailias.generation.pipeline import finalize_job, inspect_job, load_job
from ailias.storage import Store, atomic_json

REQUIRED_PLUGINS = ("PythonScriptPlugin", "MetaHumanCharacter")


def readiness(editor: Path | None, project: Path | None) -> dict:
    editor = editor or (Path(found) if (found := shutil.which("UnrealEditor-Cmd.exe")) else None)
    result = {"ready": False, "editor": str(editor) if editor else None,
              "project": str(project) if project else None, "blockers": [],
              "cloud_authentication_verified": False}
    if editor is None or not editor.is_file():
        result["blockers"].append("UnrealEditor-Cmd.exe is not installed/configured")
        return result
    editor = editor.resolve()
    if editor.name.lower() != "unrealeditor-cmd.exe" or len(editor.parents) < 3:
        result["blockers"].append("Provide the installed Engine\\Binaries\\Win64\\UnrealEditor-Cmd.exe")
        return result
    engine = editor.parents[2]
    version_path = engine / "Build" / "Build.version"
    if not version_path.is_file():
        result["blockers"].append("Engine Build.version is missing")
        return result
    version = json.loads(version_path.read_text(encoding="utf-8-sig"))
    result["version"] = ".".join(str(version[key]) for key in ("MajorVersion", "MinorVersion", "PatchVersion"))
    if (version["MajorVersion"], version["MinorVersion"]) != (5, 8):
        result["blockers"].append("Adapter targets UE 5.8 APIs; verify another engine version before use")
    if project is None or not project.is_file() or project.suffix.lower() != ".uproject":
        result["blockers"].append("An existing .uproject with the required plugins enabled is required")
        return result
    settings = json.loads(project.read_text(encoding="utf-8-sig"))
    association = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", settings.get("EngineAssociation", ""))
    if association and tuple(map(int, association.groups())) != (5, 8):
        result["blockers"].append("Project targets another engine; ailias will not silently upgrade it")
    enabled = {plugin["Name"]: plugin.get("Enabled", False) for plugin in settings.get("Plugins", [])}
    result["plugins"] = {}
    for name in REQUIRED_PLUGINS:
        manifests = list((engine / "Plugins").rglob(f"{name}.uplugin"))
        manifest = manifests[0] if len(manifests) == 1 else None
        installed = manifest is not None
        default = (json.loads(manifest.read_text(encoding="utf-8-sig")).get("EnabledByDefault") is True
                   if installed else False)
        active = enabled.get(name, default) is True
        result["plugins"][name] = {"installed": installed, "enabled": active}
        if not installed or not active:
            result["blockers"].append(f"Install/enable {name}")
    result["ready"] = not result["blockers"]
    result["editor"], result["project"] = str(editor), str(project.resolve())
    return result


def asset_path(value: str) -> str:
    if not re.fullmatch(r"/Game/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*(?:\.[A-Za-z0-9_]+)?", value):
        raise ValueError("Unreal asset must be an explicit /Game/... package/object path")
    return value


def run_unreal(store: Store, job: str, operation: str, editor: Path | None,
               project: Path | None, tools, identity=None, character=None, cloud_services=False):
    from ailias.generation import unreal_worker

    report, path = load_job(store, job)
    if inspect_job(store, job)["integrity_problems"]:
        raise ValueError("Generation inputs/artifacts changed; do not use a stale handoff")
    probe = readiness(editor, project)
    blockers = list(probe["blockers"])
    if "cleaned_mesh" not in report["artifacts"]:
        blockers.append("No cleaned, textured head candidate exists; finish reconstruction/Blender first")
    if operation == "conform" and (not identity or not character):
        raise ValueError("Conform requires an already-solved Identity and an existing MetaHuman Character")
    for value in (identity, character):
        if value is not None:
            asset_path(value)
    if operation not in {"import", "conform"}:
        raise ValueError("Unsupported Unreal operation")
    report["unreal_readiness"] = probe
    if report.get("metahuman", {}).get("status") not in {None, "not_started"}:
        report.setdefault("unreal_history", []).append(report["metahuman"])
    if blockers:
        report["metahuman"] = {"status": "blocked", "blockers": blockers,
                               "assembled": False, "rig_verified": False}
        atomic_json(path, report)
        return report
    root = path.parent / "unreal"
    root.mkdir(exist_ok=True)
    receipt = root / f"{operation}-receipt.json"
    request_path = path.parent / "unreal-request.json"
    if receipt.exists():
        raise ValueError("This Unreal operation has a receipt already; preserve it rather than overwrite")
    request = {
        "operation": operation, "receipt": str(receipt),
        "mesh": str(store.artifact_path(report["artifacts"]["cleaned_mesh"])),
        "destination": asset_path(f"/Game/Ailias/{store.root.name.replace('-', '_')}/{job.replace('-', '_')}"),
        "identity": identity, "character": character, "cloud_services": cloud_services,
    }
    atomic_json(request_path, request)
    tools.paths["unreal"] = Path(probe["editor"])
    try:
        step = tools.run("unreal", f"unreal-{operation}", [
            probe["project"], "-run=pythonscript", f"-script={Path(unreal_worker.__file__)}",
            "-unattended", "-nop4", "-nosplash", "-stdout", "-FullStdOutLogOutput",
        ], path.parent, environment={"AILIAS_UNREAL_REQUEST": str(request_path)})
        report["steps"].append(step)
        result = json.loads(receipt.read_text(encoding="utf-8"))
        allowed = {"mesh_imported", "identity_conformed", "identity_conformed_cloud_requests_returned"}
        if result.get("status") not in allowed:
            raise ValueError("Unreal did not confirm the requested operation")
        packages = result.get("assets", [result["character"]] if result.get("character") else [])
        if not packages:
            raise ValueError("Unreal receipt did not identify saved project assets")
        artifacts = {}
        for index, package in enumerate(packages):
            asset_path(package)
            relative = package.removeprefix("/Game/").split(".")[0]
            asset = Path(probe["project"]).parent / "Content" / Path(*relative.split("/"))
            saved = asset.with_suffix(".uasset")
            artifacts[f"unreal_{operation}_asset_{index}"] = store.reference_artifact(saved)
            for suffix in (".ubulk", ".uptnl", ".uexp"):
                companion = asset.with_suffix(suffix)
                if companion.is_file():
                    artifacts[f"unreal_{operation}_asset_{index}{suffix}"] = store.reference_artifact(companion)
        report["artifacts"].update(artifacts)
        report["metahuman"] = result
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["metahuman"] = {
            "status": "failed", "error": {"type": type(error).__name__, "message": str(error)},
            "project_may_contain_partial_outputs": True, "assembled": False, "rig_verified": False,
        }
    finally:
        if receipt.is_file():
            report["artifacts"][f"unreal_{operation}_receipt"] = store.reference_artifact(receipt)
        finalize_job(store, report, path)
    return report

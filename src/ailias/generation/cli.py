"""Expose native reconstruction, Blender cleanup and guarded Unreal CLI operations."""

from pathlib import Path

from ailias.generation.settings import DEFAULT_GENERATION_CONFIG, GenerationPolicy, load_settings
from ailias.generation.tools import NativeTools
from ailias.storage import Store


def add_parser(commands):
    command = commands.add_parser("generation", help="Explicit experimental COLMAP/Blender runs and Unreal handoffs")
    actions = command.add_subparsers(dest="generation_action", required=True)
    session = actions.add_parser("unreal-session", help="Start, inspect, probe or stop the owned reusable editor")
    session.add_argument("session_action", choices=("start", "status", "probe", "stop"))
    session.add_argument("--project", type=Path, required=True, help="Existing .uproject owned by this session")
    session.add_argument("--editor", type=Path, help="Installed UnrealEditor-Cmd.exe; required to start")
    session.add_argument("--wait", action="store_true", help="Keep the launcher alive until the owned editor exits")
    for name, description in (
        ("doctor", "Probe shared native tools without inference or downloads"),
        ("reconstruct", "Benchmark one curated candidate group; recover geometry, then run Blender"),
        ("select-inputs", "Audit frontal-prioritized cross-group inputs without reconstruction or GPU work"),
        ("status", "Inspect a generation job and verify retained artifacts"),
        ("clean-mesh", "Retry Blender after a successful reconstruction stage"),
    ):
        action = actions.add_parser(name, help=description)
        action.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
        if name != "doctor":
            action.add_argument("alias")
        if name in {"reconstruct", "select-inputs"}:
            action.add_argument("run")
            action.add_argument("group")
        elif name in {"status", "clean-mesh"}:
            action.add_argument("job")
    for name, description in (
        ("unreal-doctor", "Check an installed UE 5.8 editor/project; never download an engine"),
        ("unreal-import", "Import a cleaned textured mesh; this does not create a MetaHuman"),
        ("unreal-conform", "Conform an existing Character from a manually solved Identity"),
    ):
        action = actions.add_parser(name, help=description)
        action.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
        action.add_argument("--editor", type=Path, help="Installed UnrealEditor-Cmd.exe")
        action.add_argument("--project", type=Path, help="Existing .uproject; requires enabled plugins")
        if name != "unreal-doctor":
            action.add_argument("alias")
            action.add_argument("job")
        if name == "unreal-conform":
            action.add_argument("--identity", required=True, help="Already conformed /Game/... Identity asset")
            action.add_argument("--character", required=True, help="Existing /Game/... MetaHuman Character")
            action.add_argument("--cloud-services", action="store_true",
                                help="Explicitly request Epic cloud auto-rig/textures using editor authentication")


def execute(args, data_root: Path):
    if args.generation_action == "unreal-session":
        from ailias.generation.unreal_session import UnrealSession

        if args.wait and args.session_action != "start":
            raise ValueError("--wait is only supported when starting the owned editor")
        session = UnrealSession(args.project, args.editor)
        if args.session_action == "probe":
            result = session.probe(session.root / "probes")
        else:
            result = getattr(session, args.session_action)()
        if args.wait:
            session.wait()
        return result, 0

    from ailias.generation.pipeline import clean_mesh, finalize_job, inspect_job, load_job, reconstruct

    policy, tools_root = load_settings(args.generation_config)
    tools = NativeTools(tools_root, policy.timeout_seconds)
    if args.generation_action == "unreal-doctor":
        from ailias.generation.unreal import readiness
        result = readiness(args.editor, args.project)
        return result, 0 if result["ready"] else 2
    if args.generation_action == "doctor":
        result = {name: tools.probe(name) for name in ("colmap", "blender")}
        return {"tools": result, "unreal": {
            "status": "explicit_editor_and_project_required",
            "minimum_version": "5.8", "cloud_authentication": "through Epic editor; never stored by ailias",
        }}, 0 if all(probe["ready"] for probe in result.values()) else 2
    store = Store(data_root, args.alias)
    with store.lock():
        if args.generation_action == "select-inputs":
            from ailias.generation.inputs import prepare_inputs
            result = prepare_inputs(store, args.run, args.group, policy)
            return {"manifest": str(store.resolve(result["manifest"])),
                    "selection": result["selection"], "previews": result.get("previews", {})}, 0
        if args.generation_action in {"unreal-import", "unreal-conform"}:
            from ailias.generation.unreal import run_unreal
            result = run_unreal(
                store, args.job, args.generation_action.removeprefix("unreal-"),
                args.editor, args.project, tools, getattr(args, "identity", None),
                getattr(args, "character", None), getattr(args, "cloud_services", False))
            return result, 2 if result["metahuman"]["status"] in {"blocked", "failed"} else 0
        if args.generation_action == "status":
            result = inspect_job(store, args.job)
            return result, 2 if result["integrity_problems"] else 0
        if args.generation_action == "reconstruct":
            if policy.use_cleansed_portraits:
                from ailias.imageprocessing.cleansing import cleanse_run, load_policy as cleansing_policy
                from ailias.imageprocessing.matting import MattingBackend
                matting = MattingBackend(store, "cuda", policy.cpu_threads, policy.gpu_index)
                try:
                    cleanse_run(store, args.run, cleansing_policy(), matting, args.group)
                finally:
                    matting.close()
            result = reconstruct(store, args.run, args.group, tools, policy)
            return result, 0 if result["status"] == "complete" else 2
        report, path = load_job(store, args.job)
        if inspect_job(store, args.job)["integrity_problems"]:
            raise ValueError("Generation inputs/artifacts changed; do not resume stale reconstruction")
        report["status"] = "running"
        try:
            clean_mesh(store, report, path, tools, GenerationPolicy(**report["policy"]))
            report["status"] = "complete"
            report["current_step"] = "awaiting_human_mesh_and_likeness_review"
        except (OSError, ValueError, TypeError, KeyError) as error:
            report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        finally:
            finalize_job(store, report, path)
        return report, 0 if report["status"] == "complete" else 2

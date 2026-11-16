"""Expose explicit head-model setup, photo fitting and job-integrity CLI commands."""

from dataclasses import replace
from pathlib import Path

from ailias.facefitting.settings import DEFAULT_CONFIG
from ailias.generation.settings import DEFAULT_GENERATION_CONFIG
from ailias.storage import Store


def add_parser(commands):
    command = commands.add_parser("facefitting", help="Explicit dense photo-conditioned GNM head fitting")
    actions = command.add_subparsers(dest="facefitting_action", required=True)
    for name, description in (
        ("models", "Download/verify pinned public head assets and retained license provenance"),
        ("fit", "Run fresh CPU dense observations, shared shape fitting and configured Blender renders"),
        ("status", "Verify a head-fitting job and its retained source/output dependencies"),
        ("appearance", "Bake photo colour and render approximate hair on approved clay geometry"),
        ("appearance-status", "Inspect an appearance revision and its retained dependencies"),
        ("unreal-import", "Import a hair-free textured appearance head, not a finished MetaHuman"),
    ):
        action = actions.add_parser(name, help=description)
        action.add_argument("alias")
        if name == "fit":
            action.add_argument("run")
            action.add_argument("group")
            action.add_argument("--fit-config", type=Path, default=DEFAULT_CONFIG)
            action.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
        if name in {"status", "appearance", "appearance-status", "unreal-import"}:
            action.add_argument("job")
        if name == "unreal-import":
            action.add_argument("--editor", type=Path, required=True)
            action.add_argument("--project", type=Path, required=True)
            action.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
        if name == "appearance":
            action.add_argument("--appearance-config", type=Path,
                                default=DEFAULT_CONFIG.with_name("appearance.json"))
            action.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)


def execute(args, data_root):
    from ailias.facefitting.model import model_assets
    from ailias.facefitting.pipeline import inspect_fit, run_fit
    from ailias.facefitting.settings import load_policy
    from ailias.generation.settings import load_settings
    from ailias.generation.tools import NativeTools

    store = Store(data_root, args.alias)
    if args.facefitting_action == "status":
        result = inspect_fit(store, args.job)
        return result, 2 if result["integrity_problems"] else 0
    if args.facefitting_action == "appearance-status":
        from ailias.facefitting.appearance import inspect_appearance
        result = inspect_appearance(store, args.job)
        return result, 2 if result["integrity_problems"] else 0
    with store.lock():
        if args.facefitting_action == "models":
            return {"assets": model_assets(store, install=True)}, 0
        generation, tools_root = load_settings(args.generation_config)
        if args.facefitting_action == "unreal-import":
            from ailias.facefitting.unreal_handoff import import_appearance
            result = import_appearance(store, args.job, args.editor, args.project,
                                       NativeTools(tools_root, generation.timeout_seconds))
            return result, 0 if result["status"] == "static_head_imported_for_review" else 2
        if args.facefitting_action == "appearance":
            from ailias.facefitting.appearance import load_appearance_policy, run_appearance
            result = run_appearance(store, args.job, load_appearance_policy(args.appearance_config),
                                    generation, NativeTools(tools_root, generation.timeout_seconds))
            return result, 0 if result["status"] == "awaiting_appearance_review" else 2
        result = run_fit(store, args.run, args.group, load_policy(args.fit_config),
                         replace(generation, use_cleansed_portraits=True),
                         NativeTools(tools_root, generation.timeout_seconds))
        return result, 0 if result["status"] == "awaiting_human_likeness_review" else 2

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
        ("unreal-conform", "Create a new local head-only Character draft from a verified import ID"),
        ("unreal-features", "Apply scripted features to an isolated native Character revision and reload it"),
        ("unreal-review", "Render one neutral native frame from a verified feature/runtime revision"),
    ):
        action = actions.add_parser(name, help=description)
        action.add_argument("alias")
        if name == "fit":
            action.add_argument("run")
            action.add_argument("group")
            action.add_argument("--fit-config", type=Path, default=DEFAULT_CONFIG)
            action.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
        if name in {"status", "appearance", "appearance-status", "unreal-import", "unreal-conform", "unreal-review"}:
            action.add_argument("job")
        if name in {"unreal-import", "unreal-conform", "unreal-features", "unreal-review"}:
            action.add_argument("--editor", type=Path, required=True)
            action.add_argument("--project", type=Path, required=True)
            action.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
        if name == "unreal-conform":
            action.add_argument("--baseline", type=Path,
                                help="Selected Blender baseline JSON; defaults to this alias's working-baseline.json when present")
        if name == "unreal-features":
            action.add_argument("--character", required=True, help="Existing /Game/... native Character")
            action.add_argument("--profile", type=Path, required=True, help="Versioned native feature JSON")
            action.add_argument("--baseline", type=Path,
                                help="Pinned baseline JSON; defaults to this alias's working-baseline.json")
            action.add_argument("--assemble", action="store_true",
                                help="Build and independently reload an optimized medium-quality runtime actor")
            action.add_argument("--cloud-services", action="store_true",
                                help="Explicitly allow missing rig/texture requests for this isolated revision")
        if name == "unreal-review":
            from ailias.reanimation.settings import DEFAULT_CONFIG as REEL_CONFIG
            action.add_argument("--review-config", type=Path, default=REEL_CONFIG)
            action.add_argument("--resolution", default="540x960", help="Even 9:16 native review dimensions")
            action.add_argument("--diagnostic", action="store_true",
                                help="Inspect a retained failed assembly without declaring its feature checks passed")
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
        if args.facefitting_action == "unreal-review":
            from ailias.facefitting.native_review import render_review
            from ailias.reanimation.settings import load_policy as load_reel_policy
            try:
                width, height = (int(value) for value in args.resolution.lower().split("x"))
            except ValueError as error:
                raise ValueError("Review resolution must be WIDTHxHEIGHT") from error
            policy = replace(load_reel_policy(args.review_config), width=width, height=height)
            result = render_review(store, args.job, args.editor, args.project, policy,
                                   NativeTools(tools_root, policy.timeout_seconds),
                                   diagnostic=args.diagnostic)
            return result, 0 if result["status"] == "native_neutral_rendered_awaiting_visual_review" else 2
        if args.facefitting_action == "unreal-features":
            from ailias.facefitting.native_features import apply_features
            result = apply_features(store, args.character, args.profile, args.editor, args.project,
                                    NativeTools(tools_root, generation.timeout_seconds),
                                    baseline=args.baseline, assemble=args.assemble,
                                    cloud_services=args.cloud_services)
            return result, 0 if result["status"] == "features_reloaded_awaiting_visual_review" else 2
        if args.facefitting_action == "unreal-conform":
            from ailias.facefitting.unreal_conform import conform_head
            result = conform_head(store, args.job, args.editor, args.project, generation,
                                  NativeTools(tools_root, generation.timeout_seconds), baseline=args.baseline)
            return result, 0 if result["status"] == "head_conformed_awaiting_review" else 2
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

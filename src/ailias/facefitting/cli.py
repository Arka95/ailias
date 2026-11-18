"""Expose explicit head-model setup, photo fitting and job-integrity CLI commands."""

from dataclasses import replace
from pathlib import Path

from ailias.facefitting.configuration import DEFAULT_CONFIG
from ailias.generation.settings import DEFAULT_GENERATION_CONFIG
from ailias.storage import Store


def add_parser(commands):
    command = commands.add_parser("facefitting", help="Explicit dense photo-conditioned GNM head fitting")
    actions = command.add_subparsers(dest="facefitting_action", required=True)
    fresh = actions.add_parser("video-create", help="Create fresh native geometry, appearance and editor preview from sampled photos")
    fresh.add_argument("alias")
    fresh.add_argument("--evidence", type=Path, required=True,
                       help="Retained sampled-frame manifest; all downstream predictions are recomputed")
    fresh.add_argument("--video-config", type=Path, default=DEFAULT_CONFIG.with_name("video-evidence.json"),
                       help="Face detection/quality policy; video decoding is never invoked")
    fresh.add_argument("--editor", type=Path, required=True)
    fresh.add_argument("--project", type=Path, required=True)
    fresh.add_argument("--fit-config", type=Path, default=DEFAULT_CONFIG)
    fresh.add_argument("--photos-run", help="Include every accepted original-photo portrait across this scan's groups")
    fresh.add_argument("--fit-search-config", type=Path,
                       help="Explicit exhaustive identity/expression/regularization search; all candidates are retained")
    fresh.add_argument("--appearance-config", type=Path, default=DEFAULT_CONFIG.with_name("appearance.json"))
    fresh.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
    video_run = actions.add_parser("video-run", help="Run local video-to-saved-UE pipeline with timed built-in gates")
    video_run.add_argument("alias")
    video_run.add_argument("folder", type=Path)
    video_run.add_argument("--character", required=True)
    video_run.add_argument("--appearance", type=Path, required=True)
    video_run.add_argument("--correspondence", type=Path, required=True)
    video_run.add_argument("--editor", type=Path, required=True)
    video_run.add_argument("--project", type=Path, required=True)
    video_run.add_argument("--response", type=Path, help="Reuse a source-matched calibration; omit for cold end-to-end timing")
    video_run.add_argument("--video-config", type=Path, default=DEFAULT_CONFIG.with_name("video-evidence.json"))
    video_run.add_argument("--fit-config", type=Path, default=DEFAULT_CONFIG.with_name("native-video-fit.json"))
    video_run.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
    video_status = actions.add_parser("video-run-status", help="Recheck run input/output hashes without executing UE or models")
    video_status.add_argument("alias")
    video_status.add_argument("job")
    video = actions.add_parser("video-evidence", help="Extract local timestamped identity/expression frames and audio")
    video.add_argument("alias")
    video.add_argument("folder", type=Path)
    video.add_argument("--video-config", type=Path, default=DEFAULT_CONFIG.with_name("video-evidence.json"))
    video_fit = actions.add_parser("video-fit", help="Fit native shared identity with per-frame expressions; no Blender")
    video_fit.add_argument("alias")
    video_fit.add_argument("--evidence", type=Path, required=True)
    video_fit.add_argument("--response", type=Path, required=True, help="Matching native geometry-response manifest")
    video_fit.add_argument("--correspondence", type=Path, required=True, help="Native ray/barycentric NPZ with triangles")
    video_fit.add_argument("--appearance", type=Path, required=True, help="Preserved native appearance JSON")
    video_fit.add_argument("--fit-config", type=Path, default=DEFAULT_CONFIG)
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
        ("compare-features", "Compare photo/native-render evidence; optionally write a bounded iris edit proposal"),
        ("native-control-map", "Associate exported native sculpt indices with installed topology metadata"),
    ):
        action = actions.add_parser(name, help=description)
        action.add_argument("alias")
        if name == "native-control-map":
            action.add_argument("--topology", type=Path, required=True, help="Read-only exported native topology JSON")
            action.add_argument("--inspection", type=Path, required=True, help="Matching unchanged-source export receipt")
            action.add_argument("--configuration", type=Path, required=True, help="Installed landmarks_config.json")
            action.add_argument("--anatomy", type=Path, required=True, help="Installed face_landmarks.json")
            action.add_argument("--source-package", type=Path, required=True, help="Original inspected Character uasset")
        if name == "compare-features":
            action.add_argument("--reference", type=Path, required=True, help="imageprocessing feature manifest")
            action.add_argument("--reference-face-id", required=True)
            action.add_argument("--candidate", type=Path, required=True, help="inspect-image render manifest")
            action.add_argument("--current-tint", type=float, nargs=3,
                                help="Actual native RGB tint used for this render; proposal only")
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
            action.add_argument("--reference-face-id",
                                help="Use this existing fitted photo/camera for native contours; no Blender render or refit")
            action.add_argument("--source-character",
                                help="Duplicate and refine this existing native Character; preserve its original package")
        if name == "unreal-features":
            action.add_argument("--character", required=True, help="Existing /Game/... native Character")
            action.add_argument("--profile", type=Path, required=True, help="Versioned native feature JSON")
            action.add_argument("--baseline", type=Path,
                                help="Pinned baseline JSON; defaults to this alias's working-baseline.json")
            action.add_argument("--assemble", action="store_true",
                                help="Build and independently reload a runtime actor using profile assembly settings")
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
    if args.facefitting_action == "video-create":
        import json
        from ailias.facefitting.workflows.photo_to_character import run_photo_to_character_pipeline
        from ailias.facefitting.configuration import load_policy
        from ailias.facefitting.configuration import VideoPolicy
        from ailias.facefitting.configuration import load_appearance_policy
        from ailias.facefitting.geometry.identity_search import load_search_policy
        from ailias.generation.settings import load_settings
        from ailias.generation.tools import NativeTools
        store = Store(data_root, args.alias)
        generation, tools_root = load_settings(args.generation_config)
        with store.lock():
            result = run_photo_to_character_pipeline(
                store, args.evidence, args.editor, args.project,
                NativeTools(tools_root, generation.timeout_seconds), load_policy(args.fit_config),
                video_policy=VideoPolicy(**json.loads(args.video_config.read_text(encoding="utf-8"))),
                appearance_policy=load_appearance_policy(args.appearance_config),
                generation_policy=generation, photos_run=args.photos_run,
                search_policy=load_search_policy(args.fit_search_config) if args.fit_search_config else None)
        return result, 0 if result["status"] == "fresh_video_completed" else 2
    if args.facefitting_action in {"video-run", "video-run-status"}:
        import json
        from ailias.facefitting.workflows.video_character_refinement import (
            RUN_SUCCESS,
            inspect_video_refinement,
            refine_character_from_video,
        )
        store = Store(data_root, args.alias)
        if args.facefitting_action == "video-run-status":
            result = inspect_video_refinement(store, args.job)
            return result, 0 if result["status"] == RUN_SUCCESS and not result["integrity_problems"] else 2
        from ailias.facefitting.configuration import load_policy
        from ailias.facefitting.configuration import VideoPolicy
        from ailias.generation.settings import load_settings
        from ailias.generation.tools import NativeTools
        generation, tools_root = load_settings(args.generation_config)
        policy = VideoPolicy(**json.loads(args.video_config.read_text(encoding="utf-8")))
        with store.lock():
            result = refine_character_from_video(
                store, args.folder, args.character, args.appearance, args.correspondence,
                args.editor, args.project, NativeTools(tools_root, generation.timeout_seconds),
                policy, load_policy(args.fit_config), response_path=args.response)
        return result, 0 if result["status"] == RUN_SUCCESS else 2
    if args.facefitting_action == "video-fit":
        from ailias.facefitting.geometry.video_identity_fitting import fit_native_video
        from ailias.facefitting.configuration import load_policy
        store = Store(data_root, args.alias)
        with store.lock():
            result = fit_native_video(store, args.evidence, args.response, args.correspondence,
                                      args.appearance, load_policy(args.fit_config))
        return result, 0 if result["status"] == "native_video_proposal_ready" else 2
    if args.facefitting_action == "video-evidence":
        import json
        from ailias.facefitting.configuration import VideoPolicy
        from ailias.facefitting.evidence.video_sampling import extract_evidence
        policy = VideoPolicy(**json.loads(args.video_config.read_text(encoding="utf-8")))
        store = Store(data_root, args.alias)
        # Evidence decoding writes only an exclusively created UUID revision, not session state.
        result = extract_evidence(store, args.folder, policy)
        return result, 0 if result["status"] == "video_evidence_ready" else 2
    if args.facefitting_action == "native-control-map":
        from ailias.facefitting.geometry.sculpt_control_mapping import map_native_controls
        store = Store(data_root, args.alias)
        with store.lock():
            result = map_native_controls(store, args.topology, args.inspection,
                                         args.configuration, args.anatomy, args.source_package)
        return result, 0 if result["status"] == "mapped" else 2
    if args.facefitting_action == "compare-features":
        from ailias.facefitting.appearance.feature_comparison import compare_features
        store = Store(data_root, args.alias)
        with store.lock():
            return compare_features(store, args.reference, args.reference_face_id,
                                    args.candidate, args.current_tint), 0
    from ailias.facefitting.geometry.head_model import model_assets
    from ailias.facefitting.artifacts import inspect_head_fit
    from ailias.facefitting.workflows.head_fitting import fit_head_from_photos
    from ailias.facefitting.configuration import load_policy
    from ailias.generation.settings import load_settings
    from ailias.generation.tools import NativeTools

    store = Store(data_root, args.alias)
    if args.facefitting_action == "status":
        result = inspect_head_fit(store, args.job)
        return result, 2 if result["integrity_problems"] else 0
    if args.facefitting_action == "appearance-status":
        from ailias.facefitting.artifacts import inspect_head_appearance
        result = inspect_head_appearance(store, args.job)
        return result, 2 if result["integrity_problems"] else 0
    with store.lock():
        if args.facefitting_action == "models":
            return {"assets": model_assets(store, install=True)}, 0
        generation, tools_root = load_settings(args.generation_config)
        if args.facefitting_action == "unreal-review":
            from ailias.facefitting.unreal.character_portrait import render_character_portrait
            from ailias.reanimation.settings import load_policy as load_reel_policy
            try:
                width, height = (int(value) for value in args.resolution.lower().split("x"))
            except ValueError as error:
                raise ValueError("Review resolution must be WIDTHxHEIGHT") from error
            policy = replace(load_reel_policy(args.review_config), width=width, height=height)
            result = render_character_portrait(store, args.job, args.editor, args.project, policy,
                                   NativeTools(tools_root, policy.timeout_seconds),
                                   diagnostic=args.diagnostic)
            return result, 0 if result["status"] == "native_neutral_rendered_awaiting_visual_review" else 2
        if args.facefitting_action == "unreal-features":
            from ailias.facefitting.unreal.character_features import apply_character_features
            result = apply_character_features(store, args.character, args.profile, args.editor, args.project,
                                    NativeTools(tools_root, generation.timeout_seconds),
                                    baseline=args.baseline, assemble=args.assemble,
                                    cloud_services=args.cloud_services)
            return result, 0 if result["status"] == "features_reloaded_awaiting_visual_review" else 2
        if args.facefitting_action == "unreal-conform":
            from ailias.facefitting.unreal.head_conforming import conform_head
            result = conform_head(store, args.job, args.editor, args.project, generation,
                                  NativeTools(tools_root, generation.timeout_seconds), baseline=args.baseline,
                                  reference_face_id=args.reference_face_id, source_character=args.source_character)
            return result, 0 if result["status"] == "head_conformed_awaiting_review" else 2
        if args.facefitting_action == "unreal-import":
            from ailias.facefitting.unreal.static_mesh_import import import_appearance
            result = import_appearance(store, args.job, args.editor, args.project,
                                       NativeTools(tools_root, generation.timeout_seconds))
            return result, 0 if result["status"] == "static_head_imported_for_review" else 2
        if args.facefitting_action == "appearance":
            from ailias.facefitting.configuration import load_appearance_policy
            from ailias.facefitting.workflows.appearance_generation import generate_head_appearance
            result = generate_head_appearance(store, args.job, load_appearance_policy(args.appearance_config),
                                    generation, NativeTools(tools_root, generation.timeout_seconds))
            return result, 0 if result["status"] == "awaiting_appearance_review" else 2
        result = fit_head_from_photos(store, args.run, args.group, load_policy(args.fit_config),
                         replace(generation, use_cleansed_portraits=True),
                         NativeTools(tools_root, generation.timeout_seconds))
        return result, 0 if result["status"] == "awaiting_human_likeness_review" else 2

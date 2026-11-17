"""Expose explicit reel doctor, audio solve/render and evidence-status commands; never launch on import."""

from pathlib import Path

from ailias.reanimation.settings import DEFAULT_CONFIG, load_policy
from ailias.storage import Store


def add_parser(commands):
    parser = commands.add_parser("reanimation", help="Audio-driven portrait reels from existing rigged MetaHumans")
    actions = parser.add_subparsers(dest="reanimation_action", required=True)
    for name in ("doctor", "run", "status"):
        action = actions.add_parser(name)
        action.add_argument("alias")
        if name == "status":
            action.add_argument("job")
        else:
            action.add_argument("--editor", type=Path, required=True)
            action.add_argument("--project", type=Path, required=True)
            action.add_argument("--ffmpeg", type=Path)
        if name == "run":
            action.add_argument("--audio", type=Path, required=True, help="Existing mono PCM16 narration WAV")
            action.add_argument("--blueprint", help="/Game path to an assembled rigged MetaHuman Blueprint; omit for audio solve only")
            action.add_argument("--reel-config", type=Path, default=DEFAULT_CONFIG)


def execute(args, data_root):
    from ailias.reanimation.pipeline import create_reel, doctor, inspect_job

    if args.reanimation_action == "doctor":
        report = doctor(args.editor, args.project, args.ffmpeg)
        return report, 2 if report["blockers"] else 0
    store = Store(data_root, args.alias)
    if args.reanimation_action == "status":
        report = inspect_job(store, args.job)
        return report, 2 if report["integrity_problems"] else 0
    with store.lock():
        report = create_reel(store, args.audio, args.editor, args.project, load_policy(args.reel_config),
                             blueprint=args.blueprint, encoder=args.ffmpeg)
    return report, 0 if report["status"] in {
        "audio_solved_awaiting_rigged_metahuman", "video_rendered_awaiting_visual_review"} else 2

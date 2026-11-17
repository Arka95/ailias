"""Expose explicit reel doctor, audio solve/render and evidence-status commands; never launch on import."""

from dataclasses import replace
from pathlib import Path

from ailias.reanimation.settings import DEFAULT_CONFIG, load_policy
from ailias.storage import Store


def add_parser(commands):
    parser = commands.add_parser("reanimation", help="Audio-driven portrait reels from existing rigged MetaHumans")
    actions = parser.add_subparsers(dest="reanimation_action", required=True)
    for name in ("doctor", "run", "status"):
        action = actions.add_parser(name)
        action.add_argument("alias", nargs="?", help="Legacy positional alias; prefer --alias")
        action.add_argument("--alias", dest="named_alias", help="ailias character/session name")
        if name == "status":
            action.add_argument("job")
        else:
            action.add_argument("--editor", type=Path, required=True)
            action.add_argument("--project", type=Path, required=True)
            action.add_argument("--ffmpeg", type=Path)
        if name == "run":
            action.add_argument("--voice", "--audio", dest="audio", type=Path, required=True,
                                metavar="WAV", help="Supplied mono PCM16 narration WAV from any source; no voice generation")
            action.add_argument("--blueprint", help="/Game path to an assembled rigged MetaHuman Blueprint; omit for audio solve only")
            action.add_argument("--reel-config", type=Path, default=DEFAULT_CONFIG)
            action.add_argument("--resolution", metavar="WIDTHxHEIGHT",
                                help="Per-run even 9:16 dimensions, e.g. 540x960; defaults to the reel policy (1080x1920)")
            action.add_argument("--character-position", nargs=3, type=float, metavar=("X", "Y", "Z"),
                                help="Character world position in centimeters")
            action.add_argument("--character-yaw", type=float, help="Character yaw in degrees [-180,180]; camera follows its front")
            action.add_argument("--screen-position", nargs=2, type=float, metavar=("U", "V"),
                                help="Head framing target in normalized screen coordinates; top-left 0,0, center .5,.5")


def execute(args, data_root):
    from ailias.reanimation.pipeline import create_reel, doctor, inspect_job

    if args.alias and args.named_alias and args.alias != args.named_alias:
        raise ValueError("Positional alias conflicts with --alias")
    alias = args.named_alias or args.alias
    if not alias:
        raise ValueError("Specify --alias NAME (or the legacy positional alias)")
    store = Store(data_root, alias)
    if args.reanimation_action == "doctor":
        report = doctor(args.editor, args.project, args.ffmpeg)
        return report, 2 if report["blockers"] else 0
    if args.reanimation_action == "status":
        report = inspect_job(store, args.job)
        return report, 2 if report["integrity_problems"] else 0
    policy = load_policy(args.reel_config)
    overrides = {}
    if args.resolution:
        try:
            width, height = map(int, args.resolution.lower().split("x"))
        except ValueError:
            raise ValueError("Resolution must use WIDTHxHEIGHT, for example 540x960") from None
        overrides.update(width=width, height=height)
    for argument, field in (("character_position", "character_location_cm"),
                            ("character_yaw", "character_yaw_degrees"),
                            ("screen_position", "subject_screen_position")):
        value = getattr(args, argument)
        if value is not None:
            overrides[field] = value
    policy = replace(policy, **overrides)
    with store.lock():
        report = create_reel(store, args.audio, args.editor, args.project, policy,
                             blueprint=args.blueprint, encoder=args.ffmpeg)
    return report, 0 if report["status"] in {
        "audio_solved_awaiting_rigged_metahuman", "video_rendered_awaiting_visual_review"} else 2

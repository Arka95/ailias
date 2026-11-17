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
            source = action.add_mutually_exclusive_group(required=True)
            source.add_argument("--audio", type=Path, help="Existing mono PCM16 narration WAV; no synthesis")
            source.add_argument("--voice", help="Existing Yapit voice profile, e.g. testuser; requires --script")
            action.add_argument("--script", type=Path, help="Reviewed Yapit Script JSON for --voice (title and beats)")
            action.add_argument("--yapit-config", type=Path, help="Existing Yapit MCP launch settings; default config\\yapit.local.json")
            action.add_argument("--seconds", type=float, help="Optional Yapit duration target, 5-180 seconds; otherwise natural duration")
            action.add_argument("--style", default="default", help="Yapit delivery style; named styles require enrolled references")
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
    voice_selected = args.voice is not None
    if voice_selected and not args.voice.strip():
        raise ValueError("--voice must name an existing Yapit voice profile")
    if voice_selected and args.script is None:
        raise ValueError("--voice requires --script with reviewed Yapit Script JSON")
    if args.audio and (args.script is not None or args.yapit_config is not None
                       or args.seconds is not None or args.style != "default"):
        raise ValueError("--script, --yapit-config, --seconds and --style apply only to --voice, not --audio")
    if voice_selected:
        readiness = doctor(args.editor, args.project, args.ffmpeg)
        if args.blueprint and args.ffmpeg is None:
            readiness["blockers"].append("Rendering requires --ffmpeg for final H.264/AAC video")
        if readiness["blockers"]:
            return {"status": "blocked", "readiness": readiness}, 2
    with store.lock():
        options = {"blueprint": args.blueprint, "encoder": args.ffmpeg}
        audio = args.audio
        if voice_selected:
            from ailias.reanimation.narration import DEFAULT_YAPIT_CONFIG, generate_narration

            narration = generate_narration(
                store, args.voice, args.script, args.yapit_config or DEFAULT_YAPIT_CONFIG,
                seconds=args.seconds, style=args.style, max_audio_seconds=policy.max_audio_seconds)
            audio = narration["audio"]
            options["narration"] = narration
        report = create_reel(store, audio, args.editor, args.project, policy, **options)
    return report, 0 if report["status"] in {
        "audio_solved_awaiting_rigged_metahuman", "video_rendered_awaiting_visual_review"} else 2

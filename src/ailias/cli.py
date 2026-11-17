"""CLI contracts are defined in parser(); all results are JSON on stdout."""

import argparse
import json
import sys
from pathlib import Path

from ailias.intake import ingest, review_photo, validate
from ailias.settings import DEFAULT_CONFIG, load_settings
from ailias.storage import Store
from ailias.workflow import BASELINE_VIEWS, attach, baseline, status


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="ailias: manual-first photo-to-MetaHuman workflow")
    cli.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                     help="JSON configuration file")
    cli.add_argument("--data-root", type=Path, help="Override configured local data folder")
    commands = cli.add_subparsers(dest="command", required=True)
    from ailias.imageprocessing.cli import add_parser as add_imageprocessing
    from ailias.geometry.cli import add_parser as add_geometry
    from ailias.generation.cli import add_parser as add_generation
    from ailias.facefitting.cli import add_parser as add_facefitting
    from ailias.reanimation.cli import add_parser as add_reanimation
    add_imageprocessing(commands)
    add_geometry(commands)
    add_generation(commands)
    add_facefitting(commands)
    add_reanimation(commands)
    cleanup = commands.add_parser("cleanup", help="Preview or apply dependency-safe generated-resource cleanup")
    cleanup.add_argument("--alias", required=True, help="ailias character/session name")
    cleanup.add_argument("--apply", action="store_true", help="Delete planned files; default is dry-run only")
    cleanup.add_argument("--prune-job", action="append", default=[], metavar="RELATIVE_JOB",
                         help="Explicit obsolete generated job to remove, only if unchanged and unreferenced; repeatable")
    for name, help_text in (
        ("init", "Create a session without replacing existing data"),
        ("intake", "Reference and quality-check photos from a folder or selection.json"),
        ("review", "Record human-reviewed angles, expression and capture conditions"),
        ("validate", "Check photo quality and required view coverage"),
        ("attach", "Record a completed manual reconstruction/cleanup/MetaHuman handoff"),
        ("baseline", "Record a three-view visual comparison and human note"),
        ("status", "Read stage state and verify artifact integrity"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("session", help="Session identifier")
        if name == "intake":
            command.add_argument("folder", type=Path)
        elif name == "review":
            command.add_argument("photo", help="Unique SHA256 photo ID prefix")
            command.add_argument("--yaw", type=float, required=True)
            command.add_argument("--pitch", type=float, required=True)
            command.add_argument("--neutral", action=argparse.BooleanOptionalAction, required=True)
            command.add_argument("--unobstructed", action=argparse.BooleanOptionalAction, required=True)
            command.add_argument("--consistent", action=argparse.BooleanOptionalAction, required=True)
            command.add_argument("--landmarks", type=Path, help="Optional normalized landmark JSON")
        elif name == "attach":
            command.add_argument("stage", choices=("reconstruction", "cleanup", "metahuman"))
            command.add_argument("--mesh", type=Path)
            command.add_argument("--texture", type=Path)
            command.add_argument("--cameras", type=Path)
            command.add_argument("--evidence", type=Path)
            command.add_argument("--note", required=True)
        elif name == "baseline":
            for view in BASELINE_VIEWS:
                flag = view.replace("_", "-")
                command.add_argument(f"--{flag}-reference", required=True, help="Accepted photo ID prefix")
                command.add_argument(f"--{flag}-render", type=Path, required=True)
            command.add_argument("--note", required=True)
    return cli


def execute(args) -> tuple[dict, int]:
    policy, configured_root = load_settings(args.config)
    if args.command == "cleanup":
        from ailias.maintenance import cleanup_resources
        report = cleanup_resources(Store(args.data_root or configured_root, args.alias),
                                   apply=args.apply, prune_jobs=args.prune_job)
        return report, 2 if report["errors"] else 0
    if args.command in {"imageprocessing", "geometry", "generation", "facefitting", "reanimation"}:
        if args.command == "imageprocessing":
            from ailias.imageprocessing.cli import execute as run_processing
        elif args.command == "geometry":
            from ailias.geometry.cli import execute as run_processing
        elif args.command == "generation":
            from ailias.generation.cli import execute as run_processing
        elif args.command == "reanimation":
            from ailias.reanimation.cli import execute as run_processing
        else:
            from ailias.facefitting.cli import execute as run_processing
        return run_processing(args, args.data_root or configured_root)
    store = Store(args.data_root or configured_root, args.session)
    if args.command == "status":
        result = status(store)
        return result, 2 if result["integrity_problems"] else 0
    with store.lock():
        if args.command == "init":
            return store.initialize(), 0
        if args.command == "intake":
            result = ingest(store, args.folder, policy)
            return result, 2 if any("error" in photo for photo in result["photos"]) else 0
        if args.command == "review":
            landmarks = (json.loads(args.landmarks.read_text(encoding="utf-8"))
                         if args.landmarks else None)
            return review_photo(store, args.photo, args.yaw, args.pitch, args.neutral,
                                args.unobstructed, args.consistent, landmarks), 0
        if args.command == "validate":
            result = validate(store, policy)
            return result, 0 if result["ready"] else 2
        if args.command == "attach":
            artifacts = {name: getattr(args, name) for name in
                         ("mesh", "texture", "cameras", "evidence") if getattr(args, name)}
            return attach(store, policy, args.stage, artifacts, args.note), 0
        pairs = {view: (getattr(args, f"{view}_reference"),
                        getattr(args, f"{view}_render")) for view in BASELINE_VIEWS}
        return baseline(store, policy, pairs, args.note), 0


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result, code = execute(args)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}),
              file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return code

import json
import sys
from pathlib import Path

from ailias.imageprocessing.settings import DEFAULT_PROCESSING_CONFIG, load_policy
from ailias.storage import Store


def add_parser(commands):
    command = commands.add_parser("imageprocessing", help="Local photo cleansing, face aliases and body extraction")
    actions = command.add_subparsers(dest="image_action", required=True)
    models = actions.add_parser("models", help="Explicitly download SHA256-pinned CPU models into data/ALIAS")
    models.add_argument("alias")
    models.add_argument("--no-body", action="store_true", help="Install only face models")
    scan = actions.add_parser("scan", help="Analyze a file/folder locally; create reviewable face groups")
    scan.add_argument("alias")
    scan.add_argument("source", type=Path)
    scan.add_argument("--recursive", action="store_true")
    scan.add_argument("--no-body", action="store_true", help="Skip person segmentation and body masks")
    scan.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                      help="Explicit inference device; cuda fails rather than falling back to CPU")
    scan.add_argument("--limit", type=int, help="Analyze only the first N sorted input files")
    scan.add_argument("--processing-config", type=Path, default=DEFAULT_PROCESSING_CONFIG)
    select = actions.add_parser("select", help="Confirm a group for the alias and export eligible candidates")
    select.add_argument("alias")
    select.add_argument("run")
    select.add_argument("group")
    compare = actions.add_parser("compare", help="Compare two detected faces: identity versus 2D landmark shape")
    compare.add_argument("alias")
    compare.add_argument("run")
    compare.add_argument("first")
    compare.add_argument("second")
    cleanup = actions.add_parser("cleanup", help="Reclaim verified source copies and unused generated assets")
    cleanup.add_argument("alias")
    cleanup.add_argument("--prune-superseded", action="store_true",
                         help="Also remove unselected obsolete runs; requires --keep-run")
    cleanup.add_argument("--keep-run", help="Complete run to retain when pruning")
    curate = actions.add_parser("curate", help="Learn from a manually reviewed portrait folder; never deletes originals")
    curate.add_argument("alias")
    curate.add_argument("run", help="Completed extraction run whose portraits were manually reviewed")
    curate.add_argument("--removed-are-rejected", action="store_true",
                        help="Confirm kept portraits are valid and manually removed ones are invalid")
    curate.add_argument("--processing-config", type=Path, default=DEFAULT_PROCESSING_CONFIG)


def execute(args, data_root: Path):
    store = Store(data_root, args.alias)
    with store.lock():
        if args.image_action == "models":
            from ailias.imageprocessing.models import install_models
            return install_models(store, not args.no_body), 0
        if args.image_action == "scan":
            from ailias.imageprocessing.backend import OpenCVBackend
            from ailias.imageprocessing.pipeline import scan
            policy = load_policy(args.processing_config)
            if args.device == "cuda":
                from ailias.imageprocessing.cuda import OpenCVCudaBackend
                backend = OpenCVCudaBackend(store, policy, not args.no_body)
            else:
                backend = OpenCVBackend(store, policy, not args.no_body)

            def progress(event):
                if event["processed"] % 10 == 0 or event["processed"] == event["total"]:
                    print(json.dumps({"imageprocessing": event}), file=sys.stderr, flush=True)

            try:
                result = scan(store, args.source, policy, backend, args.recursive,
                              not args.no_body, args.limit, progress)
            finally:
                if hasattr(backend, "close"):
                    backend.close()
            return result, 2 if result["summary"]["errors"] else 0
        if args.image_action == "select":
            from ailias.imageprocessing.pipeline import select_group
            result = select_group(store, args.run, args.group)
            return {key: value for key, value in result.items() if key != "members"}, 0
        if args.image_action == "cleanup":
            from ailias.imageprocessing.cleanup import cleanup_alias
            result = cleanup_alias(store, args.keep_run, args.prune_superseded)
            return result, 2 if result["errors"] else 0
        if args.image_action == "curate":
            from ailias.imageprocessing.curation import capture_curation
            return capture_curation(store, args.run, load_policy(args.processing_config),
                                    args.removed_are_rejected), 0
        from ailias.geometry.comparison import compare_detected_faces
        return compare_detected_faces(store, args.run, args.first, args.second), 0

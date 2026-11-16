from pathlib import Path

from ailias.storage import Store


def add_parser(commands):
    command = commands.add_parser("geometry", help="Facial geometry errors, separate from identity similarity")
    actions = command.add_subparsers(dest="geometry_action", required=True)
    compare = actions.add_parser("compare", help="Compare corresponding 2D/3D landmarks and optional 3D surface samples")
    compare.add_argument("alias")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--fit", choices=("rigid", "similarity"), default="rigid")


def execute(args, data_root: Path):
    from ailias.geometry.comparison import compare_files
    store = Store(data_root, args.alias)
    with store.lock():
        return compare_files(store, args.reference, args.candidate, args.fit), 0

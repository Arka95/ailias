from pathlib import Path

from ailias.storage import Store


def cleanup_owned_files(store: Store, directory: str, retained: list[str]) -> dict:
    """Remove only unreferenced files inside one explicitly owned job directory."""
    relative = Path(directory)
    parts = relative.parts
    valid_scope = (
        len(parts) == 2 and parts[0] == "geometry"
        or len(parts) == 3 and parts[0] == "imageprocessing" and parts[1] in {"runs", "selections"}
    )
    if relative.is_absolute() or not valid_scope or parts[-1] in {".", ".."}:
        raise ValueError("Cleanup requires a specific relative job directory")
    root = store.resolve(directory)
    if root == store.root.resolve() or not root.exists():
        return {"removed_files": 0, "removed_bytes": 0, "errors": 0}
    keep = {store.resolve(path) for path in retained}
    result = {"removed_files": 0, "removed_bytes": 0, "errors": 0}
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.resolve() in keep:
            continue
        if not path.resolve().is_relative_to(root):
            continue
        try:
            info = path.stat()
            path.unlink()
            result["removed_files"] += 1
            result["removed_bytes"] += info.st_size if info.st_nlink <= 1 else 0
        except OSError:
            result["errors"] += 1
    for path in sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        if path.is_dir() and not path.is_symlink() and path.resolve().is_relative_to(root):
            try:
                path.rmdir()
            except OSError:
                pass
    return result

"""Finalize owned job files and enumerate cross-pipeline artifact dependencies."""

import json
import os
from pathlib import Path

from ailias.storage import Store

JOB_COLLECTIONS = (
    ("geometry",),
    ("imageprocessing", "runs"), ("imageprocessing", "selections"),
    ("imageprocessing", "cleansing"),
    ("generation", "runs"), ("generation", "inputs"),
    ("facefitting", "runs"), ("facefitting", "appearance"),
    ("facefitting", "unreal"), ("facefitting", "conforms"), ("facefitting", "features"),
    ("facefitting", "native-reviews"),
    ("facefitting", "video"), ("facefitting", "video-fits"),
    ("facefitting", "responses"), ("facefitting", "fresh-video"),
    ("facefitting", "video-runs"),
    ("reanimation", "runs"),
)


def is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def owned_files(root: Path):
    """Walk without following symlinks or Windows directory junctions."""
    if not root.is_dir() or is_link(root):
        return
    for folder, directories, files in os.walk(root, followlinks=False):
        parent = Path(folder)
        directories[:] = sorted(name for name in directories if not is_link(parent / name))
        for name in sorted(files):
            path = parent / name
            if not is_link(path) and path.is_file():
                yield path


def job_directory(store: Store, directory: str) -> Path:
    relative = Path(directory)
    if (relative.is_absolute() or ".." in relative.parts
            or not any(relative.parts[:-1] == parts for parts in JOB_COLLECTIONS)):
        raise ValueError("Cleanup requires a specific relative job directory")
    current = store.root
    for part in ("", *relative.parts):
        current = current / part
        if is_link(current):
            raise ValueError("Cleanup cannot traverse a symlink or junction")
    return store.resolve(directory)


def referenced_paths(store: Store, value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from referenced_paths(store, nested)
    elif isinstance(value, list):
        for nested in value:
            yield from referenced_paths(store, nested)
    elif isinstance(value, str) and ("\\" in value or "/" in value):
        try:
            path = Path(value)
            path = path.resolve() if path.is_absolute() else store.resolve(value)
        except (ValueError, OSError):
            return
        if path.is_relative_to(store.root.resolve()):
            yield path


def metadata_documents(store: Store):
    """Read retained metadata across every pipeline; malformed JSON fails closed."""
    paths = list(store.root.glob("*.json"))
    for module in ("imageprocessing", "geometry", "generation", "facefitting",
                   "reanimation", "apparel"):
        root = store.root / module
        for folder, directories, files in os.walk(root, followlinks=False):
            parent = Path(folder)
            if is_link(parent):
                directories[:] = []
                continue
            directories[:] = sorted(name for name in directories
                                     if name not in {"models", "work", "frames", "logs",
                                                     "worker", "review-workers", "__pycache__"}
                                     and not is_link(parent / name))
            paths.extend(parent / name for name in files if name.endswith(".json"))
    for path in sorted(set(paths)):
        if path.is_file() and not is_link(path):
            yield path, json.loads(path.read_text(encoding="utf-8-sig"))


def reference_path(store: Store, reference: dict) -> Path:
    path = Path(reference["path"])
    return path.resolve() if path.is_absolute() else store.artifact_path(reference)


def artifact_paths(store: Store, value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and "sha256" in value:
            yield reference_path(store, value)
        for nested in value.values():
            yield from artifact_paths(store, nested)
    elif isinstance(value, list):
        for nested in value:
            yield from artifact_paths(store, nested)


def cleanup_owned_files(store: Store, directory: str, retained: list[str]) -> dict:
    """Remove only unreferenced files inside one explicitly owned job directory."""
    root = job_directory(store, directory)
    if root == store.root.resolve() or not root.exists():
        return {"removed_files": 0, "removed_bytes": 0, "errors": 0}
    keep = {store.resolve(path) for path in retained}
    result = {"removed_files": 0, "removed_bytes": 0, "errors": 0}
    for path in owned_files(root):
        if path.resolve() in keep or not path.resolve().is_relative_to(root):
            continue
        try:
            info = path.stat()
            path.unlink()
            result["removed_files"] += 1
            result["removed_bytes"] += info.st_size if info.st_nlink <= 1 else 0
        except OSError:
            result["errors"] += 1
    remove_empty_directories(root)
    return result


def remove_empty_directories(root: Path):
    children = []
    for folder, directories, _ in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not is_link(Path(folder) / name)]
        children.extend(Path(folder) / name for name in directories)
    for path in reversed(children):
        if not is_link(path) and path.resolve().is_relative_to(root.resolve()):
            try:
                path.rmdir()
            except OSError:
                pass


def finalize_resources(store: Store, report: dict, manifest: Path, retained=(), *,
                       retain_metadata=True) -> dict:
    """Remove uncommitted outputs from this invocation, retaining results and diagnostics."""
    root = manifest.parent
    relative = str(root.relative_to(store.root))
    result = {"removed_files": 0, "removed_bytes": 0, "errors": 0}
    keep = {manifest.resolve()}
    try:
        keep.update(artifact_paths(store, report))
        keep.update(Path(path).resolve() for path in retained)
        files = list(owned_files(root))
        keep.update(path.resolve() for path in files
                    if path.suffix == ".log" or retain_metadata and path.suffix == ".json")
        result = cleanup_owned_files(store, relative, [
            str(path.relative_to(store.root.resolve())) for path in keep
            if path.is_relative_to(store.root.resolve())
        ])
    except (OSError, ValueError):
        result["errors"] += 1
    if report.get("status") == "running":
        report["status"] = "interrupted"
    report["cleanup"] = result
    report["resource_ownership"] = {
        "version": 1, "directory": relative,
        "retained": [],
    }
    for path in owned_files(root):
        try:
            if (path.resolve() == manifest.resolve() or path.resolve() not in keep
                    or not path.stat().st_size):
                continue
            report["resource_ownership"]["retained"].append(store.reference_artifact(path))
        except OSError:
            result["errors"] += 1
    return result

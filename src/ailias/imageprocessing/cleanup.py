import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from ailias.resources import cleanup_owned_files
from ailias.storage import Store, atomic_json, digest


def _documents(store: Store):
    paths = [store.path, store.resolve(str(Path("imageprocessing") / "selection.json")),
             store.resolve(str(Path("imageprocessing") / "curation.json"))]
    for collection in ("runs", "selections", "comparisons", "cleansing"):
        root = store.resolve(str(Path("imageprocessing") / collection))
        if root.exists():
            paths.extend(root.rglob("*.json"))
    geometry = store.resolve("geometry")
    if geometry.exists():
        paths.extend(geometry.rglob("*.json"))
    generation = store.resolve("generation")
    for collection, filename in (("runs", "manifest.json"), ("inputs", "sources.json")):
        paths.extend((generation / collection).glob(f"*\\{filename}"))
    for path in paths:
        if path.is_file():
            # Unknown or malformed metadata is not safe to garbage-collect around.
            yield path, json.loads(path.read_text(encoding="utf-8"))


def _references(store: Store, value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _references(store, nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _references(store, nested)
    elif isinstance(value, str) and ("\\" in value or "/" in value):
        try:
            path = Path(value)
            path = path.resolve() if path.is_absolute() else store.resolve(value)
        except (ValueError, OSError):
            return
        if path.is_relative_to(store.root.resolve()):
            yield path


def cleanup_alias(store: Store, keep_run: str | None = None,
                  prune_superseded: bool = False) -> dict:
    """Migrate safe source references and reclaim only verifiably owned redundancy."""
    root = store.resolve(str(Path("imageprocessing") / "runs"))
    documents = list(_documents(store))
    manifests = {
        path.parent.name: (path, report) for path, report in documents
        if path.name == "manifest.json" and path.parent.parent == root
        and report.get("schema_version") == 1 and report.get("run") == path.parent.name
    }
    if prune_superseded and (
        keep_run not in manifests or manifests[keep_run][1].get("status") != "complete"
    ):
        raise ValueError("Pruning requires an existing complete --keep-run")
    result = {"pruned_runs": [], "migrated_sources": 0, "reused_generated_assets": 0,
              "removed_files": 0, "removed_bytes": 0, "retained_missing_sources": 0,
              "retained_empty_runs": [], "errors": 0}
    protected = {keep_run} if keep_run else set()
    for path, value in documents:
        references = set(_references(store, value))
        for run, (manifest, _) in manifests.items():
            if path == manifest:
                continue
            directory = root / run
            if (isinstance(value, dict) and value.get("run") == run
                    or any(ref.is_relative_to(directory) for ref in references)):
                protected.add(run)
    if prune_superseded:
        for run, (path, report) in list(manifests.items()):
            if run in protected or report.get("status") not in {"complete", "failed"}:
                continue
            directory = str(path.parent.relative_to(store.root))
            removed = cleanup_owned_files(store, directory, [])
            for key in ("removed_files", "removed_bytes", "errors"):
                result[key] += removed[key]
            try:
                path.parent.rmdir()
            except OSError:
                pass
            if not path.exists():
                result["pruned_runs"].append(run)
                del manifests[run]
    for run, (manifest, report) in manifests.items():
        if report.get("status") not in {"complete", "failed"}:
            continue
        changed = False
        for source in report.get("sources", {}).values():
            artifact = source["artifact"]
            if artifact.get("storage") == "external":
                continue
            original = store.artifact_path(artifact)
            originals = manifest.parent / "originals"
            if not original.is_relative_to(originals):
                continue
            replacement = None
            for filename in source.get("input_files", []):
                candidate = Path(filename)
                if (candidate.is_absolute() and candidate.resolve() != original
                        and candidate.is_file() and digest(candidate) == artifact["sha256"]):
                    replacement = store.reference_artifact(candidate)
                    break
            if replacement:
                source["artifact"] = replacement
                result["migrated_sources"] += 1
                changed = True
            else:
                result["retained_missing_sources"] += 1
        for face in report.get("faces", {}).values():
            for role, relative in list(face["products"].items()):
                path = store.resolve(relative)
                if not path.is_relative_to(manifest.parent) or path.suffix.lower() != ".png":
                    continue
                checksum = digest(path)
                category = {"portrait": "portraits", "body": "bodies", "mask": "masks"}[role]
                target_relative = Path("imageprocessing") / "assets" / category / f"{checksum}.png"
                target = store.resolve(str(target_relative))
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if digest(target) != checksum:
                        raise ValueError("Existing generated asset failed verification during migration")
                else:
                    # Same-volume hard link publishes a canonical asset without copying pixels.
                    os.link(path, target)
                face["products"][role] = str(target_relative)
                result["reused_generated_assets"] += 1
                changed = True
        if changed:
            report["source_storage"] = (
                "reference-only" if all(source["artifact"].get("storage") == "external"
                                        for source in report["sources"].values()) else "mixed"
            )
            report["generated_storage"] = "alias-content-addressed-assets-v1"
            report["storage_migrated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(manifest, report)
    # Re-read after migration so no file still referenced by selections/history is deleted.
    references = {
        path for _, value in _documents(store) for path in _references(store, value)
    }
    retained = [str(path.relative_to(store.root.resolve())) for path in references]
    for _, (manifest, report) in manifests.items():
        if report.get("status") not in {"complete", "failed"}:
            continue
        own = str(manifest.parent.relative_to(store.root))
        removed = cleanup_owned_files(store, own, [*retained, str(manifest.relative_to(store.root))])
        for key in ("removed_files", "removed_bytes", "errors"):
            result[key] += removed[key]
    assets = store.resolve(str(Path("imageprocessing") / "assets"))
    if assets.exists():
        for category in ("portraits", "review", "bodies", "masks", "cleansed", "cutouts", "alphas", "aligned"):
            folder = assets / category
            if not folder.exists():
                continue
            for path in folder.iterdir():
                if (path.is_file() and not path.is_symlink() and path.resolve() not in references
                        and re.fullmatch(r"[a-f0-9]{64}\.png", path.name)):
                    size = path.stat().st_size
                    path.unlink()
                    result["removed_files"] += 1
                    result["removed_bytes"] += size
    if root.exists():
        for directory in root.iterdir():
            if (directory.is_dir() and not directory.is_symlink()
                    and re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{8}", directory.name)):
                try:
                    directory.rmdir()
                except OSError:
                    if not any(directory.iterdir()):
                        result["retained_empty_runs"].append(directory.name)
    return result

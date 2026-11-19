"""Retain completed major-stage outputs and their live dependencies, not every attempt."""

from contextlib import nullcontext
from datetime import datetime, timezone
import json
from contextvars import ContextVar
from pathlib import Path
import uuid

from ailias.resource_management.job_resources import (
    is_link,
    owned_files,
    remove_empty_directories,
)
from ailias.storage import atomic_json
from ailias.resource_management.scratch_cleanup import _candidate_path, _remnant
from ailias.common.json_values import iter_artifact_references
from ailias.resource_management.catalog import ACTIVE, SUCCESS, ArtifactCatalog
from ailias.common.file_checksums import FileChecksums

_running = ContextVar("major_retention_running", default=frozenset())


def plan_major_cleanup(store, *, native=None, checksums=None):
    digest = checksums if checksums is not None else FileChecksums()
    if not store.root.is_dir() or is_link(store.root):
        raise ValueError("Cleanup requires an existing unlinked alias")
    catalog = ArtifactCatalog(store, digest)
    jobs = catalog.discover_jobs()
    roots, keep, documents = catalog.select_live_outputs(jobs)
    known, job_files = {}, {}
    for root, (manifest, report) in jobs.items():
        files = list(owned_files(root))
        job_files[root] = files
        for ref in iter_artifact_references(report):
            path = catalog.resolve_reference(ref)
            known.setdefault(path, set()).add(ref["sha256"])
    for document in documents.values():
        for ref in iter_artifact_references(document):
            path = catalog.resolve_reference(ref)
            known.setdefault(path, set()).add(ref["sha256"])
    native_plan = None
    if native is not None:
        native_plan = native.plan(keep, documents, known, checksums=digest)
        keep.update(Path(path) for path in native_plan["kept_files"])
    candidates, obsolete, protected_changes = [], [], []
    for root, (manifest, report) in jobs.items():
        if report.get("status") in ACTIVE:
            continue
        obsolete_job = manifest not in keep
        if obsolete_job:
            obsolete.append(
                {
                    "manifest": str(manifest.relative_to(store.root)),
                    "sha256": digest(manifest),
                    "already_pruned": report.get("status") == "pruned",
                }
            )
        for path in job_files[root]:
            if path in keep or path == manifest:
                continue
            registered = path in known
            remnant = _remnant(path, root, report)
            if not (obsolete_job and registered or remnant):
                continue
            if remnant in {
                "encoded_video_frame",
                "cancelled_render_frame",
            } and not store.verify(report["artifacts"]["video"]):
                continue
            checksum = digest(path)
            if registered and checksum not in known[path]:
                protected_changes.append(str(path.relative_to(store.root)))
                continue
            stat = path.stat()
            candidates.append(
                {
                    "path": str(path.relative_to(store.root)),
                    "sha256": checksum,
                    "bytes": stat.st_size,
                    "reclaimable_bytes": stat.st_size if stat.st_nlink == 1 else 0,
                    "reason": remnant or "obsolete_registered_output",
                }
            )
    return {
        "policy": "final_major_stage_outputs",
        "dry_run": True,
        "kept_major_outputs": sorted(
            str(path.relative_to(store.root)) for path in roots
        ),
        "kept_files": sorted(str(path) for path in keep),
        "metadata": [
            {"path": str(path.relative_to(store.root)), "sha256": digest(path)}
            for path in documents
            if path.is_relative_to(store.root)
        ],
        "job_metadata": [
            {"path": str(manifest.relative_to(store.root)), "sha256": digest(manifest)}
            for manifest, _ in jobs.values()
        ],
        "protected": [
            {"path": str(path.relative_to(store.root)), "sha256": digest(path)}
            for path in keep
            if path.is_file()
            and path.is_relative_to(store.root)
            and path.suffix not in {".uasset", ".umap"}
        ],
        "jobs": obsolete,
        "candidates": candidates,
        "native": native_plan,
        "protected_changed_files": protected_changes,
        "planned_files": len(candidates)
        + len(native_plan["delete"] if native_plan else []),
        "planned_bytes": sum(item["reclaimable_bytes"] for item in candidates)
        + (native_plan["planned_bytes"] if native_plan else 0),
        "errors": 0,
    }


def validate_retention_inputs(store, plan, checksums=None):
    """Reject stale plans, including newly published selectors and review dependencies."""
    checksums = checksums if checksums is not None else FileChecksums()
    catalog = ArtifactCatalog(store, checksums)
    jobs = catalog.discover_jobs()
    roots, keep, documents = catalog.select_live_outputs(jobs)
    if {str(manifest.relative_to(store.root)) for manifest, _ in jobs.values()} != {
        item["path"] for item in plan["job_metadata"]
    }:
        raise ValueError("Job metadata changed; create a fresh plan")
    if (
        sorted(str(path.relative_to(store.root)) for path in roots)
        != plan["kept_major_outputs"]
    ):
        raise ValueError("Major outputs changed; create a fresh plan")
    if {
        str(path.relative_to(store.root))
        for path in documents
        if path.is_relative_to(store.root)
    } != {item["path"] for item in plan["metadata"]}:
        raise ValueError("Live metadata changed; create a fresh plan")
    if keep - {Path(path) for path in plan["kept_files"]}:
        raise ValueError("Live dependencies changed; create a fresh plan")
    for item in [
        *plan["metadata"],
        *plan["job_metadata"],
        *plan["protected"],
        *plan["jobs"],
        *plan["candidates"],
    ]:
        relative = item.get("manifest", item.get("path"))
        raw = store.root / relative
        path = store.resolve(relative)
        if is_link(raw) or not path.is_file() or checksums(path) != item["sha256"]:
            raise ValueError("Retention inputs changed; create a fresh plan")
        if any(
            is_link(parent)
            for parent in raw.parents
            if parent.is_relative_to(store.root)
        ):
            raise ValueError("Retention cannot traverse links")
    for item in plan["candidates"]:
        _candidate_path(store, item)


def apply_major_cleanup(store, plan, *, native=None, checksums=None):
    """Preflight every exact file; archive lineage before removing superseded outputs."""
    checksums = checksums if checksums is not None else FileChecksums()
    validate_retention_inputs(store, plan, checksums)
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    archive = store.resolve(str(Path("maintenance") / "retention" / stamp))
    archive.mkdir(parents=True, exist_ok=False)
    for item in plan["jobs"]:
        if not item["already_pruned"]:
            source = store.resolve(item["manifest"])
            destination = archive / "lineage" / item["manifest"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    result = {
        "policy": plan["policy"],
        "dry_run": False,
        "removed_files": 0,
        "removed_bytes": 0,
        "errors": 0,
    }
    atomic_json(archive / "plan.json", plan)
    if plan["native"] is not None:
        if native is None:
            raise ValueError("Native retention requires the inspected owned editor")
        native_result = native.apply(plan["native"])
        result["native"] = native_result
        result["removed_files"] += native_result["removed_files"]
        result["removed_bytes"] += native_result["removed_bytes"]
        result["errors"] += native_result["errors"]
        atomic_json(archive / "native-result.json", native_result)
        validate_retention_inputs(store, plan, checksums)
    for item in plan["candidates"]:
        path = store.resolve(item["path"])
        try:
            path.unlink()
            result["removed_files"] += 1
            result["removed_bytes"] += item["reclaimable_bytes"]
        except OSError:
            result["errors"] += 1
    catalog = ArtifactCatalog(store, checksums)
    for item in plan["jobs"]:
        manifest = store.resolve(item["manifest"])
        original = json.loads(manifest.read_text(encoding="utf-8-sig"))
        retained = {
            str(path): ref
            for ref in iter_artifact_references(original)
            if (path := catalog.resolve_reference(ref)).is_file()
            and checksums(path) == ref["sha256"]
        }
        lineage = original.get("retention", {}).get("lineage")
        if not item["already_pruned"]:
            lineage = str(
                (archive / "lineage" / item["manifest"]).relative_to(store.root)
            )
        pruned = {
            "revision": original.get("revision", manifest.parent.name),
            "status": "pruned",
            "artifacts": {
                f"retained_{index}": ref for index, ref in enumerate(retained.values())
            },
            "retention": {"policy": plan["policy"], "lineage": lineage},
        }
        if pruned != original:
            atomic_json(manifest, pruned)
            remove_empty_directories(manifest.parent)
    result["receipt"] = str((archive / "receipt.json").relative_to(store.root))
    atomic_json(archive / "receipt.json", result)
    return result


def cleanup_major_outputs(store, *, apply=False, native=None):
    if not store.root.is_dir() or is_link(store.root):
        raise ValueError("Cleanup requires an existing unlinked alias")
    context = nullcontext() if getattr(store, "lock_owned", False) else store.lock()
    with context:
        checksums = FileChecksums()
        plan = plan_major_cleanup(store, native=native, checksums=checksums)
        return (
            apply_major_cleanup(store, plan, native=native, checksums=checksums)
            if apply
            else plan
        )


def finalize_major_output(store, report, manifest):
    if store.root in _running.get() or report.get("status") not in SUCCESS:
        return
    token = _running.set(_running.get() | {store.root})
    try:
        atomic_json(manifest, report)
        from ailias.generation.unreal_retention import available_native_retention

        native = available_native_retention(store, report)
        result = cleanup_major_outputs(store, apply=True, native=native)
        report["retention"] = {
            key: result[key]
            for key in ("policy", "removed_files", "removed_bytes", "errors", "receipt")
        }
        report["retention"]["native_deferred"] = native is None or bool(
            result.get("native", {}).get("deferred_files")
        )
    except (OSError, ValueError, KeyError) as error:
        report["retention"] = {
            "policy": "final_major_stage_outputs",
            "deferred": True,
            "error_type": type(error).__name__,
        }
    finally:
        _running.reset(token)

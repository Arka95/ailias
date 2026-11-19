"""Dry-run-first reclamation of job remnants; never purge inputs, models or UE packages."""

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ailias.resource_management.job_resources import (
    JOB_COLLECTIONS,
    artifact_paths,
    is_link,
    job_directory,
    metadata_documents,
    owned_files,
    reference_path,
    referenced_paths,
    remove_empty_directories,
)
from ailias.storage import Store, atomic_json, digest
from ailias.common.json_values import iter_artifact_references, iter_nested_strings

ACTIVE_STATUSES = {"running", "pending", "processing", "interrupted"}
JOB_ID = re.compile(r"(?:\d{8}T\d{6}Z-[a-f0-9]{8}|[a-f0-9]{32}|[a-f0-9]{64})")


def _remnant(path, root, report):
    relative = path.relative_to(root)
    if "work" in relative.parts or path.suffix in {".tmp", ".download"}:
        return "temporary_work"
    if path.name == "tracking-rgba8.bin" or re.fullmatch(r"photo-\d+\.npz", path.name):
        return "temporary_native_input"
    if root.parent.parent.name == "reanimation" and not report.get(
        "keep_frames", False
    ):
        if relative.parts[0] == "frames" and re.fullmatch(
            r"frame\.\d+\.png", path.name
        ):
            if report.get("video_verified") is True and "video" in report.get(
                "artifacts", {}
            ):
                return "encoded_video_frame"
            if str(report.get("status", "")).startswith(
                "cancelled"
            ) and "video" in report.get("artifacts", {}):
                return "cancelled_render_frame"
            if report.get("status") == "failed" or str(
                report.get("status", "")
            ).startswith("cancelled"):
                return "failed_render_frame"
        if (
            path.name == "portrait-reel.mp4"
            and report.get("status") == "failed"
            and "video" not in report.get("artifacts", {})
        ):
            return "partial_video"
    return None


def plan_cleanup(store: Store, prune_jobs=()):
    documents = list(metadata_documents(store))
    snapshots = [
        {"path": str(path.relative_to(store.root)), "sha256": digest(path)}
        for path, _ in documents
    ]
    jobs = {}
    for collection in JOB_COLLECTIONS:
        parent = store.root.joinpath(*collection)
        if not parent.is_dir() or is_link(parent):
            continue
        for child in parent.iterdir():
            if not child.is_dir() or is_link(child) or not JOB_ID.fullmatch(child.name):
                continue
            metadata = [
                (path, value)
                for path, value in documents
                if path.parent == child
                and path.name in {"manifest.json", "sources.json", "report.json"}
            ]
            if metadata:
                jobs[child.resolve()] = metadata[0]
    pruned = {job_directory(store, name) for name in prune_jobs}
    for root in pruned:
        if root not in jobs:
            raise ValueError(f"No recognized generated job metadata: {root.name}")
        if jobs[root][1].get("status") in ACTIVE_STATUSES:
            raise ValueError(f"Cannot prune active/interrupted job: {root.name}")
    retained_documents = [
        (path, value)
        for path, value in documents
        if not any(path.is_relative_to(root) for root in pruned)
    ]
    document_references = [
        (path, set(referenced_paths(store, value)))
        for path, value in retained_documents
    ]
    references = {ref for _, refs in document_references for ref in refs}
    strong_references = {
        path for _, value in retained_documents for path in artifact_paths(store, value)
    }
    for root in pruned:
        if any(path.is_relative_to(root) for path in references):
            raise ValueError(f"Job is referenced by retained metadata: {root.name}")
        for _, value in retained_documents:
            if root.name in set(iter_nested_strings(value)):
                raise ValueError(
                    f"Job ID is referenced by retained metadata: {root.name}"
                )
    candidates = []
    for root, (manifest, report) in sorted(jobs.items()):
        if report.get("status") in ACTIVE_STATUSES:
            continue
        files = list(owned_files(root))
        if any(
            is_link(Path(folder) / name)
            for folder, directories, filenames in os.walk(root, followlinks=False)
            for name in (*directories, *filenames)
        ):
            if root in pruned:
                raise ValueError(f"Linked resources prevent pruning: {root.name}")
            continue
        job_documents = [
            (path, value) for path, value in documents if path.is_relative_to(root)
        ]
        own_refs = {
            reference_path(store, ref): ref
            for _, value in job_documents
            for ref in iter_artifact_references(value)
        }
        if root in pruned:
            owned = set(own_refs)
            owned.update(path for path, _ in job_documents)
            owned.update(
                path
                for path in files
                if path.suffix == ".log"
                or path.parent.name in {"worker", "review-workers"}
            )
            for path in files:
                if path not in owned and not _remnant(path, root, report):
                    raise ValueError(f"Unregistered file prevents pruning: {path.name}")
            for path, ref in own_refs.items():
                if path.is_relative_to(root) and (
                    not path.is_file() or digest(path) != ref["sha256"]
                ):
                    raise ValueError(
                        f"Changed/missing output prevents pruning: {path.name}"
                    )
        protected = strong_references | {
            ref
            for path, refs in document_references
            if not path.is_relative_to(root)
            for ref in refs
        }
        frames = sorted(
            path
            for path in files
            if path.parent == root / "frames"
            and re.fullmatch(r"frame\.\d+\.png", path.name)
        )
        if frames and root not in pruned:
            protected.update((frames[0], frames[-1]))
        video = report.get("artifacts", {}).get("video")
        video_unchanged = bool(video and store.verify(video))
        for path in files:
            reason = (
                "explicitly_pruned_job"
                if root in pruned
                else _remnant(path, root, report)
            )
            if reason is None or path.resolve() in protected:
                continue
            if (
                reason in {"encoded_video_frame", "cancelled_render_frame"}
                and not video_unchanged
            ):
                continue
            stat = path.stat()
            candidates.append(
                {
                    "path": str(path.relative_to(store.root)),
                    "sha256": digest(path),
                    "bytes": stat.st_size,
                    "reclaimable_bytes": stat.st_size if stat.st_nlink <= 1 else 0,
                    "reason": reason,
                }
            )
    return {
        "dry_run": True,
        "candidates": candidates,
        "planned_files": len(candidates),
        "planned_bytes": sum(item["reclaimable_bytes"] for item in candidates),
        "prune_jobs": sorted(str(path.relative_to(store.root)) for path in pruned),
        "pruned_jobs": [],
        "metadata": snapshots,
        "removed_files": 0,
        "removed_bytes": 0,
        "errors": 0,
    }


def _candidate_path(store, item):
    parts = Path(item["path"]).parts
    collection = next(
        (
            parts[: len(scope) + 1]
            for scope in JOB_COLLECTIONS
            if parts[: len(scope)] == scope and len(parts) > len(scope) + 1
        ),
        None,
    )
    if collection is None:
        raise ValueError("Cleanup candidate is outside a generated job")
    root = job_directory(store, str(Path(*collection)))
    path = store.root / item["path"]
    if any(
        is_link(parent)
        for parent in (path, *path.parents)
        if parent.is_relative_to(root)
    ):
        raise ValueError("Cleanup candidate traverses a symlink or junction")
    path = store.resolve(item["path"])
    if not path.is_relative_to(root) or digest(path) != item["sha256"]:
        raise ValueError("Cleanup candidate changed; create a fresh plan")
    return root, path


def apply_cleanup(store: Store, plan):
    """Revalidate the complete plan before deleting any exact, owned file."""
    for item in plan["metadata"]:
        path = store.resolve(item["path"])
        if not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError("Cleanup metadata changed; create a fresh plan")
    current_metadata = {
        str(path.relative_to(store.root)) for path, _ in metadata_documents(store)
    }
    if current_metadata != {item["path"] for item in plan["metadata"]}:
        raise ValueError("Cleanup metadata set changed; create a fresh plan")
    roots = {_candidate_path(store, item)[0] for item in plan["candidates"]}
    result = {**plan, "dry_run": False, "pruned_jobs": []}
    for item in plan["candidates"]:
        try:
            _, path = _candidate_path(store, item)
            path.unlink()
            result["removed_files"] += 1
            result["removed_bytes"] += item["reclaimable_bytes"]
        except (OSError, ValueError):
            result["errors"] += 1
    for root in roots:
        remove_empty_directories(root)
        relative = str(root.relative_to(store.root))
        if relative in plan["prune_jobs"]:
            try:
                root.rmdir()
                result["pruned_jobs"].append(relative)
            except OSError:
                result["errors"] += 1
    return result


def cleanup_resources(store: Store, *, apply=False, prune_jobs=()):
    if not store.root.is_dir() or is_link(store.root):
        raise ValueError("Cleanup requires an existing, non-linked alias directory")
    with store.lock():
        plan = plan_cleanup(store, prune_jobs)
        if not apply:
            return plan
        result = apply_cleanup(store, plan)
        stamp = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        receipt = store.resolve(str(Path("maintenance") / "cleanups" / f"{stamp}.json"))
        atomic_json(receipt, result)
        return {**result, "receipt": str(receipt)}

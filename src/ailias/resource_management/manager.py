"""Coordinate resource ownership; domain modules never choose deletion policy."""

from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

from ailias.resource_management.job_resources import (
    finalize_resources,
    is_link,
    referenced_paths,
)
from ailias.storage import atomic_json

_deferred_retention = ContextVar("deferred_retention", default=frozenset())


class ResourceManager:
    def __init__(self, store, native=None):
        self.store, self.native = store, native

    @contextmanager
    def defer_retention(self):
        """Child jobs release scratch immediately; the pipeline retains major outputs once."""
        token = _deferred_retention.set(_deferred_retention.get() | {self.store.root})
        try:
            yield
        finally:
            _deferred_retention.reset(token)

    def cleanup(self, *, apply=False, prune_jobs=(), scratch_only=False):
        from ailias.resource_management.job_resources import is_link

        if not self.store.root.is_dir() or is_link(self.store.root):
            raise ValueError("Cleanup requires an existing unlinked alias")
        if scratch_only or prune_jobs:
            from ailias.resource_management.scratch_cleanup import cleanup_resources

            return cleanup_resources(self.store, apply=apply, prune_jobs=prune_jobs)
        from ailias.generation.unreal_retention import available_native_retention
        from ailias.resource_management.retention import cleanup_major_outputs

        native = (
            self.native
            if self.native is not None
            else available_native_retention(self.store)
        )
        return cleanup_major_outputs(self.store, apply=apply, native=native)

    def cleanup_unreferenced_image_assets(self, candidates, retained):
        """Release unchanged, owned image assets only when no retained metadata references them."""
        from ailias.resource_management.catalog import ArtifactCatalog
        from ailias.storage import digest

        keep = {self.store.resolve(path) for path in retained}
        candidates = [
            path for path in candidates if self.store.resolve(path) not in keep
        ]
        result = {"removed_files": 0, "removed_bytes": 0, "errors": 0}
        if not candidates:
            return result
        catalog = ArtifactCatalog(self.store)
        jobs = catalog.discover_jobs()
        _, _, documents = catalog.select_live_outputs(jobs, verify_outputs=False)
        for value in [*(report for _, report in jobs.values()), *documents.values()]:
            keep.update(referenced_paths(self.store, value))
        assets = self.store.root / "imageprocessing" / "assets"
        for relative in candidates:
            raw = self.store.root / relative
            path = self.store.resolve(relative)
            if (
                path in keep
                or not path.is_relative_to(assets)
                or is_link(raw)
                or any(
                    is_link(parent)
                    for parent in raw.parents
                    if parent.is_relative_to(self.store.root)
                )
            ):
                continue
            try:
                if not path.is_file() or digest(path) != path.stem:
                    continue
                stat = path.stat()
                path.unlink()
                result["removed_files"] += 1
                result["removed_bytes"] += stat.st_size if stat.st_nlink == 1 else 0
            except OSError:
                result["errors"] += 1
        return result

    cleanup_created_assets = cleanup_unreferenced_image_assets

    def finalize_job(
        self,
        report,
        manifest,
        retained=(),
        *,
        retain_metadata=True,
        created_assets=(),
        automatic_retention=True,
    ):
        if self.native_operation_pending(report):
            report["cleanup"] = {
                "removed_files": 0,
                "removed_bytes": 0,
                "errors": 0,
                "deferred": "Native operation has not confirmed completion",
            }
            atomic_json(manifest, report)
            return report["cleanup"]
        result = finalize_resources(
            self.store, report, manifest, retained, retain_metadata=retain_metadata
        )
        if created_assets:
            try:
                released = self.cleanup_unreferenced_image_assets(
                    created_assets,
                    [
                        str(Path(path).resolve().relative_to(self.store.root))
                        for path in retained
                        if Path(path).resolve().is_relative_to(self.store.root)
                    ],
                )
                for key, value in released.items():
                    result[key] += value
            except (OSError, ValueError):
                result["errors"] += 1
        atomic_json(manifest, report)
        from ailias.resource_management.retention import finalize_major_output

        if automatic_retention and self.store.root not in _deferred_retention.get():
            finalize_major_output(self.store, report, manifest)
        atomic_json(manifest, report)
        return result

    def native_operation_pending(self, report):
        if report.get("project_may_contain_partial_outputs") and report.get(
            "error", {}
        ).get("type") in {"TimeoutError", "TimeoutExpired"}:
            return True
        project = report.get("readiness", {}).get("project")
        if not project or not Path(project).is_file():
            return False
        from ailias.resource_management.catalog import SUCCESS

        if report.get("status") in SUCCESS and not report.get(
            "project_may_contain_partial_outputs"
        ):
            return False
        from ailias.generation.unreal_session import UnrealSession

        try:
            return UnrealSession(project).status()["status"] not in {
                "idle",
                "completed",
                "failed",
                "stopped",
            }
        except (OSError, ValueError):
            return True


def finalize_job_resources(
    store,
    report,
    manifest,
    retained=(),
    *,
    retain_metadata=True,
    created_assets=(),
    automatic_retention=True,
):
    return ResourceManager(store).finalize_job(
        report,
        manifest,
        retained,
        retain_metadata=retain_metadata,
        created_assets=created_assets,
        automatic_retention=automatic_retention,
    )

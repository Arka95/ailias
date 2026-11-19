"""Discover generated jobs and resolve the dependency closure of live outputs."""

import json
import os
from pathlib import Path

from ailias.common.json_values import iter_artifact_references, iter_nested_strings
from ailias.common.file_checksums import FileChecksums
from ailias.resource_management.job_resources import (
    JOB_COLLECTIONS,
    is_link,
    reference_path,
)
from ailias.resource_management.scratch_cleanup import JOB_ID

SUCCESS = {
    "complete",
    "awaiting_human_likeness_review",
    "awaiting_appearance_review",
    "static_head_imported_for_review",
    "head_conformed_awaiting_review",
    "features_reloaded_awaiting_visual_review",
    "native_neutral_rendered_awaiting_visual_review",
    "video_evidence_ready",
    "native_response_measured_awaiting_geometry_fit",
    "fresh_video_completed",
    "photo_video_completed_awaiting_visual_review",
    "video_rendered_awaiting_visual_review",
    "audio_solved_awaiting_rigged_metahuman",
    "native_video_completed_awaiting_likeness_review",
    "native_video_proposal_ready",
}
ACTIVE = {"running", "pending", "processing", "interrupted"}


class ArtifactCatalog:
    def __init__(self, store, checksums=None):
        self.store = store
        self._resolved = {}
        self.checksums = checksums if checksums is not None else FileChecksums()

    def resolve_reference(self, reference):
        key = (reference["path"], reference.get("storage"))
        if key not in self._resolved:
            self._resolved[key] = reference_path(self.store, reference)
        return self._resolved[key]

    def referenced_files(self, value):
        for reference in iter_artifact_references(value):
            yield self.resolve_reference(reference)

    def discover_jobs(self):
        jobs = {}
        for collection in JOB_COLLECTIONS:
            parent = self.store.root.joinpath(*collection)
            if not parent.is_dir() or is_link(parent):
                continue
            for directory in parent.iterdir():
                if (
                    not directory.is_dir()
                    or is_link(directory)
                    or not JOB_ID.fullmatch(directory.name)
                ):
                    continue
                manifest = next(
                    (
                        directory / name
                        for name in (
                            "manifest.json",
                            "sources.json",
                            "report.json",
                            "selection.json",
                        )
                        if (directory / name).is_file()
                    ),
                    None,
                )
                if manifest is not None:
                    jobs[directory] = (
                        manifest,
                        json.loads(manifest.read_text(encoding="utf-8-sig")),
                    )
        return jobs

    def outputs_intact(self, report):
        try:
            return all(
                (path := self.resolve_reference(reference)).is_file()
                and self.checksums(path) == reference["sha256"]
                for reference in iter_artifact_references(report.get("artifacts", {}))
            )
        except OSError:
            return False

    def select_live_outputs(self, jobs, *, verify_outputs=True):
        completed, roots = {}, set()
        for root, (manifest, report) in jobs.items():
            status = report.get("status")
            if status in ACTIVE:
                roots.add(manifest)
            if status not in SUCCESS:
                continue
            if status == "features_reloaded_awaiting_visual_review" and not report.get(
                "fresh_editor_reload_verified"
            ):
                continue
            if status in {
                "video_rendered_awaiting_visual_review",
                "photo_video_completed_awaiting_visual_review",
            } and not report.get("video_verified"):
                continue
            project = report.get("readiness", {}).get("project", "")
            key = (root.parent, os.path.normcase(str(project)))
            completed.setdefault(key, []).append((manifest, report))
        for versions in completed.values():
            versions.sort(
                key=lambda version: version[0].stat().st_mtime_ns, reverse=True
            )
            roots.add(versions[0][0])
            if verify_outputs:
                for manifest, report in versions:
                    if self.outputs_intact(report):
                        roots.add(manifest)
                        break
        keep, documents, pending = set(), {}, list(roots)
        session = self.store.root / "session.json"
        if session.is_file():
            state = json.loads(session.read_text(encoding="utf-8-sig"))
            documents[session] = {
                key: state[key] for key in ("photos", "stages") if key in state
            }
            pending.extend(self.referenced_files(documents[session]))
            keep.add(session)
        canonical = list(self.store.root.glob("*.json"))
        for module in (
            "imageprocessing",
            "geometry",
            "generation",
            "facefitting",
            "reanimation",
            "apparel",
        ):
            canonical.extend((self.store.root / module).glob("*.json"))
        canonical.extend((self.store.root / "facefitting" / "reviews").glob("*.json"))
        for path in canonical:
            if path == session or is_link(path):
                continue
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            documents[path] = value
            keep.add(path)
            pending.extend(self.referenced_files(value))
            for text in iter_nested_strings(value):
                if not text.endswith(".json") or not ("\\" in text or "/" in text):
                    continue
                try:
                    reference = self.store.resolve(text)
                except ValueError:
                    continue
                if reference.is_file():
                    pending.append(reference)
        while pending:
            path = Path(pending.pop())
            if path in keep:
                continue
            keep.add(path)
            if (
                not path.is_relative_to(self.store.root)
                or path.suffix.lower() != ".json"
                or not path.is_file()
            ):
                continue
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            documents[path] = value
            pending.extend(self.referenced_files(value))
        return roots, keep, documents

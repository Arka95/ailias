"""Exercise owned cleanup, dependency protection and dry-run/application behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ailias.maintenance import apply_cleanup, cleanup_resources, plan_cleanup
from ailias.resources import finalize_resources, job_directory
from ailias.storage import Store, atomic_json


class ResourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / "data", "subject")

    def write(self, name, content=b"generated"):
        path = self.store.resolve(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def job(self, collection="facefitting\\runs", status="failed"):
        root = self.store.resolve(str(Path(collection) / ("a"*32)))
        root.mkdir(parents=True, exist_ok=True)
        path = root / "manifest.json"
        report = {"revision": root.name, "status": status, "artifacts": {}}
        atomic_json(path, report)
        return root, path, report

    def test_native_finalizers_remove_partial_outputs_but_keep_results_and_failure_evidence(self):
        for collection in ("facefitting\\runs", "facefitting\\appearance", "facefitting\\unreal",
                           "facefitting\\conforms", "reanimation\\runs"):
            with self.subTest(collection=collection):
                root, manifest, report = self.job(collection)
                keep = root / "head.blend"
                keep.write_bytes(b"committed output")
                report["artifacts"]["scene"] = self.store.reference_artifact(keep)
                partial = root / "partial.blend"
                partial.write_bytes(b"incomplete native output")
                scratch = root / "work" / "partial.tmp"
                scratch.parent.mkdir()
                scratch.write_bytes(b"scratch")
                evidence = root / "receipt.json"
                evidence.write_text('{"status":"failed"}')
                log = root / "native.log"
                log.write_text("failure evidence")
                finalize_resources(self.store, report, manifest)
                self.assertFalse(partial.exists())
                self.assertFalse(scratch.parent.exists())
                self.assertTrue(keep.exists())
                self.assertTrue(evidence.exists())
                self.assertTrue(log.exists())
                self.assertEqual(report["cleanup"]["errors"], 0)
                self.assertTrue(all(self.store.verify(ref)
                                    for ref in report["resource_ownership"]["retained"]))

    def test_root_and_parent_traversal_are_never_valid_cleanup_scopes(self):
        for name in (".", "facefitting", "reanimation\\runs", "models\\a",
                     "facefitting\\conforms\\..", "generation\\runs\\..\\models"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                job_directory(self.store, name)

    def test_unknown_alias_dry_run_does_not_create_new_resource_directories(self):
        with self.assertRaisesRegex(ValueError, "existing"):
            cleanup_resources(self.store)
        self.assertFalse(self.store.root.exists())

    def reel(self, keep=False):
        root, manifest, report = self.job("reanimation\\runs", "video_rendered_awaiting_visual_review")
        video = root / "portrait-reel.mp4"
        video.write_bytes(b"verified video")
        frames = root / "frames"
        frames.mkdir()
        images = []
        for number in range(4):
            path = frames / f"frame.{number:06d}.png"
            path.write_bytes(b"frame data")
            images.append(path)
        report.update(video_verified=True, keep_frames=keep)
        report["artifacts"] = {"video": self.store.reference_artifact(video),
                               "first_frame": self.store.reference_artifact(images[0]),
                               "last_frame": self.store.reference_artifact(images[-1])}
        atomic_json(manifest, report)
        return root, manifest, report, images

    def test_dry_run_changes_nothing_and_apply_removes_only_redundant_frames(self):
        _, manifest, report, images = self.reel()
        preview = cleanup_resources(self.store)
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["planned_files"], 2)
        self.assertTrue(all(path.exists() for path in images))
        result = cleanup_resources(self.store, apply=True)
        self.assertEqual(result["removed_files"], 2)
        self.assertTrue(images[0].exists() and images[-1].exists())
        self.assertFalse(images[1].exists() or images[2].exists())
        self.assertTrue(all(self.store.verify(ref) for ref in report["artifacts"].values()))
        self.assertTrue(manifest.exists())
        self.assertEqual(cleanup_resources(self.store)["planned_files"], 0)

    def test_frame_retention_and_changed_video_both_prevent_reclamation(self):
        _, manifest, report, _ = self.reel(True)
        self.assertEqual(cleanup_resources(self.store)["planned_files"], 0)
        report["keep_frames"] = False
        self.store.artifact_path(report["artifacts"]["video"]).write_bytes(b"edited video")
        atomic_json(manifest, report)
        self.assertEqual(cleanup_resources(self.store)["planned_files"], 0)

    def test_historical_failed_frames_keep_boundary_diagnostics_and_drop_partial_video(self):
        root, manifest, report, images = self.reel()
        report.update(status="failed", video_verified=False)
        report["artifacts"].pop("video")
        atomic_json(manifest, report)
        result = cleanup_resources(self.store, apply=True)
        self.assertEqual(result["removed_files"], 3)
        self.assertTrue(images[0].exists() and images[-1].exists())
        self.assertFalse((root / "portrait-reel.mp4").exists())

    def test_explicitly_cancelled_revision_drops_frames_but_preserves_the_existing_video(self):
        _, manifest, report, images = self.reel()
        report.update(status="cancelled_for_lighting_revision", video_verified=False)
        atomic_json(manifest, report)
        result = cleanup_resources(self.store, apply=True)
        self.assertEqual(result["removed_files"], 2)
        self.assertTrue(images[0].exists() and images[-1].exists())
        self.assertTrue(self.store.verify(report["artifacts"]["video"]))

    def test_linked_candidate_parent_invalidates_a_plan_without_deleting_files(self):
        root, _, _, images = self.reel()
        plan = plan_cleanup(self.store)
        with patch("ailias.resource_management.scratch_cleanup.is_link", side_effect=lambda path: path == root / "frames"):
            with self.assertRaisesRegex(ValueError, "symlink or junction"):
                apply_cleanup(self.store, plan)
        self.assertTrue(all(path.exists() for path in images))

    def test_cross_pipeline_reference_protects_an_interior_frame(self):
        _, _, _, images = self.reel()
        atomic_json(self.store.resolve("facefitting\\reviews\\review.json"),
                    {"reference": self.store.reference_artifact(images[1])})
        plan = cleanup_resources(self.store)
        self.assertEqual([item["path"] for item in plan["candidates"]],
                         [str(images[2].relative_to(self.store.root))])

    def test_legacy_absolute_audit_references_are_resolved_without_external_storage_flag(self):
        _, _, _, images = self.reel()
        reference = self.store.reference_artifact(images[1])
        reference["path"] = str(images[1])
        atomic_json(self.store.resolve("reanimation\\migration-audit.json"), {"source": reference})
        plan = cleanup_resources(self.store)
        self.assertEqual(len(plan["candidates"]), 1)
        self.assertEqual(plan["candidates"][0]["path"], str(images[2].relative_to(self.store.root)))

    def test_own_request_does_not_pin_disposable_native_buffers(self):
        root, _, _ = self.job("facefitting\\conforms")
        buffer = root / "tracking-rgba8.bin"
        buffer.write_bytes(b"temporary pixels")
        atomic_json(root / "conform-request.json", {"rgba": str(buffer)})
        result = cleanup_resources(self.store, apply=True)
        self.assertEqual(result["removed_files"], 1)
        self.assertFalse(buffer.exists())
        self.assertTrue((root / "conform-request.json").exists())

    def test_running_jobs_and_external_sources_are_untouched(self):
        root, _, _ = self.job(status="running")
        temporary = root / "unfinished.tmp"
        temporary.write_bytes(b"active")
        original = self.write("originals\\photo.jpg")
        result = cleanup_resources(self.store, apply=True)
        self.assertEqual(result["removed_files"], 0)
        self.assertTrue(temporary.exists() and original.exists())

    def test_malformed_metadata_aborts_before_removing_any_file(self):
        root, _, _ = self.job()
        scratch = root / "orphan.tmp"
        scratch.write_bytes(b"scratch")
        self.write("facefitting\\reviews\\broken.json", b"{bad")
        with self.assertRaises(ValueError):
            cleanup_resources(self.store, apply=True)
        self.assertTrue(scratch.exists())

    def test_created_assets_cleanup_preserves_retained_shared_modified_and_unowned_files(self):
        import hashlib
        from ailias.resource_management import ResourceManager

        created = []
        for value in (b"unused", b"retained", b"shared", b"changed"):
            relative = str(Path("imageprocessing") / "assets" / "portraits" /
                           f"{hashlib.sha256(value).hexdigest()}.png")
            self.write(relative, value)
            created.append(relative)
        self.store.resolve(created[-1]).write_bytes(b"owner modified")
        _, manifest, report = self.job()
        report["shared_image"] = created[2]
        atomic_json(manifest, report)
        unowned = self.write("originals\\photo.png")
        result = ResourceManager(self.store).cleanup_created_assets(
            [*created, str(unowned.relative_to(self.store.root))], [created[1]]
        )
        self.assertEqual(result["removed_files"], 1)
        self.assertFalse(self.store.resolve(created[0]).exists())
        self.assertTrue(all(self.store.resolve(path).exists() for path in created[1:]))
        self.assertTrue(unowned.exists())

    def test_child_scratch_finalizes_but_major_retention_waits_for_pipeline_boundary(self):
        from ailias.resource_management import ResourceManager

        manager = ResourceManager(self.store)
        root, manifest, report = self.job(status="complete")
        scratch = root / "temporary.bin"
        scratch.write_bytes(b"scratch")
        with patch("ailias.resource_management.retention.finalize_major_output") as major:
            with manager.defer_retention():
                with manager.defer_retention():
                    manager.finalize_job(report, manifest)
                self.assertFalse(scratch.exists())
                major.assert_not_called()
            manager.finalize_job(report, manifest)
            major.assert_called_once()

    def test_wrapped_native_timeout_preserves_files_while_editor_still_writes(self):
        from ailias.resource_management import ResourceManager

        root, manifest, report = self.job()
        scratch = root / "in-progress.bin"
        scratch.write_bytes(b"still being produced")
        project = self.write("unreal\\Review\\Review.uproject", b"{}")
        report.update(readiness={"project": str(project)},
                      error={"type": "ToolError", "message": "native timeout"})
        for state in ("busy", "pending", "starting"):
            with self.subTest(state=state), \
                    patch("ailias.generation.unreal_session.UnrealSession.status",
                          return_value={"status": state}):
                result = ResourceManager(self.store).finalize_job(report, manifest)
                self.assertIn("deferred", result)
                self.assertTrue(scratch.exists())
        with patch("ailias.generation.unreal_session.UnrealSession.status",
                   return_value={"status": "completed"}):
            result = ResourceManager(self.store).finalize_job(report, manifest)
        self.assertNotIn("deferred", result)
        self.assertFalse(scratch.exists())

    def test_explicit_prune_refuses_approved_baselines_and_changed_outputs(self):
        root, manifest, report = self.job()
        scene = root / "approved.blend"
        scene.write_bytes(b"approved")
        report["artifacts"]["scene"] = self.store.reference_artifact(scene)
        atomic_json(manifest, report)
        baseline = self.store.resolve("facefitting\\working-baseline.json")
        atomic_json(baseline, {"artifacts": {"scene": report["artifacts"]["scene"]}})
        relative = str(root.relative_to(self.store.root))
        with self.assertRaisesRegex(ValueError, "referenced"):
            cleanup_resources(self.store, apply=True, prune_jobs=[relative])
        baseline.unlink()
        scene.write_bytes(b"owner edited")
        with self.assertRaisesRegex(ValueError, "Changed"):
            cleanup_resources(self.store, apply=True, prune_jobs=[relative])
        self.assertEqual(scene.read_bytes(), b"owner edited")

    def test_explicit_prune_preserves_unregistered_owner_files(self):
        root, _, _ = self.job()
        owner = root / "owner-edited.blend"
        owner.write_bytes(b"owner work")
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            cleanup_resources(self.store, apply=True,
                              prune_jobs=[str(root.relative_to(self.store.root))])
        self.assertTrue(owner.exists())

    def test_explicit_prune_removes_only_an_unchanged_unreferenced_job(self):
        root, manifest, report = self.job()
        output = root / "old.blend"
        output.write_bytes(b"obsolete generated output")
        report["artifacts"]["scene"] = self.store.reference_artifact(output)
        atomic_json(manifest, report)
        relative = str(root.relative_to(self.store.root))
        result = cleanup_resources(self.store, apply=True, prune_jobs=[relative])
        self.assertEqual(result["removed_files"], 2)
        self.assertFalse(root.exists())
        self.assertTrue(self.store.root.is_dir())

    def test_changed_candidate_or_new_reference_invalidates_a_plan(self):
        root, _, _ = self.job()
        scratch = root / "orphan.tmp"
        scratch.write_bytes(b"scratch")
        plan = plan_cleanup(self.store)
        scratch.write_bytes(b"owner replaced contents")
        with self.assertRaises(ValueError):
            apply_cleanup(self.store, plan)
        plan = plan_cleanup(self.store)
        atomic_json(self.store.resolve("facefitting\\reviews\\new.json"), {"source": str(scratch)})
        with self.assertRaises(ValueError):
            apply_cleanup(self.store, plan)
        self.assertTrue(scratch.exists())

    def test_image_cleanup_protects_assets_used_only_by_downstream_face_work(self):
        from ailias.imageprocessing.cleanup import cleanup_alias

        image = self.write("imageprocessing\\assets\\portraits\\" + "b"*64 + ".png")
        atomic_json(self.store.resolve("facefitting\\reviews\\retained.json"),
                    {"source": self.store.reference_artifact(image)})
        cleanup_alias(self.store)
        self.assertTrue(image.exists())

    def test_cli_cleanup_defaults_to_dry_run(self):
        from ailias.cli import execute, parser

        root, _, _ = self.job()
        scratch = root / "orphan.tmp"
        scratch.write_bytes(b"scratch")
        args = parser().parse_args(["--data-root", str(self.store.root.parent),
                                    "cleanup", "--alias", "subject"])
        report, code = execute(args)
        self.assertEqual(code, 0)
        self.assertTrue(report["dry_run"] and scratch.exists())

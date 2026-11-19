"""Major-output retention protects dependencies, edits and concurrent publication."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from ailias.resource_management.retention import (
    apply_major_cleanup,
    plan_major_cleanup,
    cleanup_major_outputs,
)
from ailias.storage import Store, atomic_json


class RetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name), "subject")
        self.sequence = 0

    def job(self, collection="facefitting\\runs", status="complete", dependencies=()):
        self.sequence += 1
        root = self.store.resolve(str(Path(collection) / f"{self.sequence:032x}"))
        root.mkdir(parents=True)
        output = root / "head.obj"
        output.write_bytes(f"head-{self.sequence}".encode())
        manifest = root / "manifest.json"
        report = {
            "revision": root.name,
            "status": status,
            "artifacts": {"mesh": self.store.reference_artifact(output)},
            "input_dependencies": list(dependencies),
        }
        atomic_json(manifest, report)
        os.utime(
            manifest, ns=(self.sequence * 1_000_000_000, self.sequence * 1_000_000_000)
        )
        return manifest, output

    def test_only_latest_completed_stage_and_actual_cross_stage_dependencies_survive(
        self,
    ):
        old, old_mesh = self.job()
        _, mesh = self.job()
        other, _ = self.job(
            "generation\\runs", dependencies=[self.store.reference_artifact(old_mesh)]
        )
        discarded, discarded_mesh = self.job("generation\\runs", status="failed")
        plan = plan_major_cleanup(self.store)
        self.assertNotIn(
            str(old_mesh.relative_to(self.store.root)),
            {item["path"] for item in plan["candidates"]},
        )
        result = apply_major_cleanup(self.store, plan)
        self.assertTrue(old_mesh.exists() and mesh.exists() and other.exists())
        self.assertFalse(discarded_mesh.exists())
        self.assertEqual(json.loads(discarded.read_text())["status"], "pruned")
        self.assertGreater(result["removed_bytes"], 0)
        self.assertTrue(list((self.store.root / "maintenance").rglob("receipt.json")))

    def test_failed_attempt_cannot_replace_last_completed_output(self):
        good, mesh = self.job()
        self.job(status="failed")
        plan = plan_major_cleanup(self.store)
        self.assertIn(
            str(good.relative_to(self.store.root)), plan["kept_major_outputs"]
        )
        apply_major_cleanup(self.store, plan)
        self.assertTrue(mesh.exists())

    def test_corrupt_declared_success_cannot_evict_last_intact_major_output(self):
        good, good_mesh = self.job()
        newest, newest_mesh = self.job()
        newest_mesh.write_bytes(b"corrupted after successful publication")
        plan = plan_major_cleanup(self.store)
        self.assertIn(
            str(good.relative_to(self.store.root)), plan["kept_major_outputs"]
        )
        self.assertIn(
            str(newest.relative_to(self.store.root)), plan["kept_major_outputs"]
        )
        apply_major_cleanup(self.store, plan)
        self.assertTrue(good_mesh.exists())
        self.assertTrue(newest_mesh.exists())

    def test_changed_and_unregistered_owner_files_are_not_deleted(self):
        manifest, mesh = self.job()
        mesh.write_bytes(b"owner modified")
        unknown = manifest.parent / "owner-notes.txt"
        unknown.write_text("owner work")
        self.job()
        plan = plan_major_cleanup(self.store)
        self.assertIn(
            str(mesh.relative_to(self.store.root)), plan["protected_changed_files"]
        )
        apply_major_cleanup(self.store, plan)
        self.assertTrue(mesh.exists() and unknown.exists())

    def test_metadata_change_or_new_job_aborts_before_any_deletion(self):
        for change in ("edit", "new"):
            with self.subTest(change=change):
                old, mesh = self.job()
                self.job()
                plan = plan_major_cleanup(self.store)
                if change == "edit":
                    value = json.loads(old.read_text())
                    value["note"] = "changed"
                    atomic_json(old, value)
                else:
                    self.job()
                with self.assertRaises(ValueError):
                    apply_major_cleanup(self.store, plan)
                self.assertTrue(mesh.exists())

    def test_retained_dependency_change_aborts_before_any_deletion(self):
        _, old_mesh = self.job()
        current, current_mesh = self.job()
        plan = plan_major_cleanup(self.store)
        current_mesh.write_bytes(b"modified after planning")
        with self.assertRaises(ValueError):
            apply_major_cleanup(self.store, plan)
        self.assertTrue(old_mesh.exists())

    def test_active_output_and_unknown_directory_are_not_pruned(self):
        active, mesh = self.job(status="running")
        self.job()
        unknown = self.store.resolve("facefitting\\runs\\owner-work")
        unknown.mkdir()
        atomic_json(unknown / "manifest.json", {"status": "failed"})
        plan = plan_major_cleanup(self.store)
        apply_major_cleanup(self.store, plan)
        self.assertTrue(mesh.exists() and active.exists())
        self.assertTrue((unknown / "manifest.json").exists())

    def test_repeat_cleanup_has_no_more_payloads_to_remove(self):
        self.job()
        self.job()
        apply_major_cleanup(self.store, plan_major_cleanup(self.store))
        again = plan_major_cleanup(self.store)
        self.assertEqual(again["planned_files"], 0)

    def test_repeat_cleanup_does_not_rewrite_unchanged_tombstones(self):
        old, _ = self.job()
        self.job()
        apply_major_cleanup(self.store, plan_major_cleanup(self.store))
        os.utime(old, ns=(1_000_000_000, 1_000_000_000))
        before = old.stat().st_mtime_ns
        apply_major_cleanup(self.store, plan_major_cleanup(self.store))
        self.assertEqual(old.stat().st_mtime_ns, before)

    def test_cleanup_reuses_only_the_calling_threads_owned_lock(self):
        self.job()
        with self.store.lock():
            self.assertTrue(cleanup_major_outputs(self.store)["dry_run"])
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(cleanup_major_outputs, self.store)
                with self.assertRaisesRegex(ValueError, "locked"):
                    result.result(timeout=5)
            self.assertTrue(self.store.lock_owned)
            with self.assertRaisesRegex(ValueError, "locked"):
                with self.store.lock():
                    self.fail("An unrelated acquisition must not become reentrant")
        self.assertFalse(self.store.lock_owned)
        with self.store.lock():
            self.assertTrue(self.store.lock_owned)

    def test_legacy_absolute_review_source_is_not_a_managed_artifact(self):
        _, old_mesh = self.job()
        self.job()
        external = Path(self.store.root.parent) / "original.jpg"
        external.write_bytes(b"original")
        atomic_json(
            self.store.root / "review.json", {"path": str(external), "sha256": "0" * 64}
        )
        apply_major_cleanup(self.store, plan_major_cleanup(self.store))
        self.assertTrue(external.exists())
        self.assertFalse(old_mesh.exists())

    def test_new_review_dependency_invalidates_plan(self):
        _, old_mesh = self.job()
        self.job()
        plan = plan_major_cleanup(self.store)
        reviews = self.store.root / "facefitting" / "reviews"
        atomic_json(reviews / "approval.json", self.store.reference_artifact(old_mesh))
        with self.assertRaises(ValueError):
            apply_major_cleanup(self.store, plan)
        self.assertTrue(old_mesh.exists())

    def test_publication_during_native_cleanup_preserves_generic_outputs(self):
        _, old_mesh = self.job()
        self.job()
        plan = plan_major_cleanup(self.store)
        plan["native"] = {"test": True}
        store = self.store

        class PublishingNative:
            def apply(self, _):
                atomic_json(
                    store.root / "review.json", store.reference_artifact(old_mesh)
                )
                return {"removed_files": 0, "removed_bytes": 0, "errors": 0}

        with self.assertRaises(ValueError):
            apply_major_cleanup(self.store, plan, native=PublishingNative())
        self.assertTrue(old_mesh.exists())
        self.assertTrue(list((store.root / "maintenance").rglob("native-result.json")))

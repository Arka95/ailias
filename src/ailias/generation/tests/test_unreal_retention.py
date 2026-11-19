"""Verify native retention without starting an editor or loading native assets."""

import json
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from ailias.generation.unreal_retention import NativeRetention
from ailias.generation.unreal_session import UnrealSession
from ailias.storage import Store


class NativeRetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name), "subject")
        project = self.store.resolve("unreal\\Review\\Review.uproject")
        project.parent.mkdir(parents=True)
        project.write_text("{}")
        self.session = UnrealSession(project)
        self.native = NativeRetention(self.store, self.session)
        self.obsolete = project.parent / "Content" / "Ailias" / "Old.uasset"
        self.obsolete.parent.mkdir(parents=True)
        self.obsolete.write_bytes(b"old")
        self.current = self.obsolete.with_name("Current.uasset")
        self.current.write_bytes(b"current")
        self.inspection = {
            "graph": {
                "/Game/Ailias/Old": {"dependencies": [], "referencers": []},
                "/Game/Ailias/Current": {"dependencies": [], "referencers": []},
            },
            "dirty_packages": [],
        }
        self.known = {
            path: {self.store.reference_artifact(path)["sha256"]}
            for path in (self.obsolete, self.current)
        }

    def plan(self):
        with patch.object(self.native, "inspection", return_value=self.inspection):
            return self.native.plan({self.current}, {}, self.known)

    def test_running_editor_queues_without_native_deletion_or_shutdown(self):
        plan = self.plan()
        with (
            patch.object(self.session, "status", return_value={"status": "idle"}),
            patch.object(self.native, "run") as run,
        ):
            result = self.native.apply(plan)
        self.assertEqual(result["deferred_files"], 1)
        self.assertTrue(self.obsolete.exists())
        self.assertEqual(json.loads(self.native.pending.read_text())["plan"], plan)
        run.assert_not_called()

    @patch("ailias.generation.unreal_retention.subprocess.run")
    def test_offline_cleanup_removes_exact_package_and_preserves_live_output(
        self, processes
    ):
        processes.return_value = Mock(stdout="")
        result = self.native.apply(self.plan())
        self.assertEqual(result["removed_files"], 1)
        self.assertFalse(self.obsolete.exists())
        self.assertTrue(self.current.exists())

    @patch("ailias.generation.unreal_retention.subprocess.run")
    def test_changed_or_new_package_aborts_before_any_deletion(self, processes):
        processes.return_value = Mock(stdout="")
        for change in ("edit", "new"):
            with self.subTest(change=change):
                plan = self.plan()
                if change == "edit":
                    self.current.write_bytes(b"owner modified")
                else:
                    self.obsolete.with_name("New.uasset").write_bytes(b"new")
                with self.assertRaises(ValueError):
                    self.native.apply(plan)
                self.assertTrue(self.obsolete.exists())
                self.current.write_bytes(b"current")

    @patch("ailias.generation.unreal_retention.subprocess.run")
    def test_borrowed_editor_or_unknown_process_prevents_offline_deletion(
        self, processes
    ):
        plan = self.plan()
        for command in ('UnrealEditor.exe "C:\\elsewhere\\Review.uproject"', None):
            processes.return_value = Mock(
                stdout=json.dumps({"ProcessId": 123, "CommandLine": command})
            )
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.native.apply(plan)
            self.assertTrue(self.obsolete.exists())

    def test_dirty_and_retained_references_are_dependency_roots(self):
        for protection in ("dirty", "reference"):
            with self.subTest(protection=protection):
                if protection == "dirty":
                    self.inspection["dirty_packages"] = ["/Game/Ailias/Old"]
                else:
                    self.inspection["dirty_packages"] = []
                    self.inspection["graph"]["/Game/Ailias/Current"]["dependencies"] = [
                        "/Game/Ailias/Old"
                    ]
                    self.inspection["graph"]["/Game/Ailias/Old"]["referencers"] = [
                        "/Game/Ailias/Current"
                    ]
                self.assertEqual(self.plan()["delete"], [])

    @patch("ailias.generation.unreal_retention.subprocess.run")
    def test_intervening_writer_is_detected_after_acquiring_the_deletion_lock(
        self, processes
    ):
        processes.return_value = Mock(stdout="")
        plan = self.plan()

        @contextmanager
        def intervening_writer():
            self.obsolete.write_bytes(b"modified immediately before lock acquisition")
            yield

        with patch.object(self.session, "lock", side_effect=intervening_writer):
            with self.assertRaisesRegex(ValueError, "content changed"):
                self.native.apply(plan)
        self.assertEqual(
            self.obsolete.read_bytes(),
            b"modified immediately before lock acquisition",
        )
        self.assertTrue(self.current.exists())

    @patch("ailias.generation.unreal_retention.subprocess.run")
    def test_queued_snapshot_can_be_replanned_after_natural_shutdown(self, processes):
        processes.return_value = Mock(stdout="")
        with patch.object(self.session, "status", return_value={"status": "idle"}):
            self.native.apply(self.plan())
        result = self.native.plan({self.current}, {}, self.known)
        self.native.apply(result)
        self.assertFalse(self.obsolete.exists())
        self.assertTrue(self.current.exists())

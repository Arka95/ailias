"""Verify ownership, project serialization and persistent routing without launching Unreal."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ailias.generation.cli import execute
from ailias.generation.tools import NativeTools, ToolError
from ailias.generation.unreal_session import UnrealSession, execute_unreal, load_settings, process_identity
from ailias.storage import atomic_json


class UnrealSessionTests(unittest.TestCase):
    def test_invalid_session_settings_are_rejected(self):
        baseline = {"mode": "persistent", "startup_timeout_seconds": 180, "poll_seconds": .1}
        for overrides in ({"mode": "unknown"}, {"startup_timeout_seconds": 0},
                          {"poll_seconds": True}, {"poll_seconds": float("inf")}, {"extra": 1}):
            with self.subTest(overrides=overrides), \
                    patch("ailias.generation.unreal_session.CONFIG") as configuration:
                configuration.read_text.return_value = json.dumps({**baseline, **overrides})
                with self.assertRaises(ValueError):
                    load_settings()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "Review.uproject"
        self.project.write_text("{}")
        self.editor = self.root / "Editor.exe"
        self.editor.write_bytes(b"fixture")
        self.session = UnrealSession(self.project, self.editor)
        self.identity = {"pid": 42, "created": 123, "executable": os.path.normcase(str(self.editor))}
        atomic_json(self.session.metadata, {
            "session_id": "owned", "project": str(self.project), "process": self.identity,
            "worker_sha256": "fixture",
        })
        atomic_json(self.session.state, {"session_id": "owned", "status": "idle"})

    def test_project_writer_lock_is_shared_across_coordinators_and_released_after_exception(self):
        other = UnrealSession(self.project, self.editor)
        with self.session.lock():
            with self.assertRaisesRegex(ValueError, "busy"):
                with other.lock():
                    self.fail("Concurrent project writer was admitted")
        with self.assertRaises(RuntimeError), other.lock():
            raise RuntimeError("failed caller")
        with self.session.lock():
            pass

    def test_recycled_pid_and_stale_bootstrap_state_do_not_establish_ownership(self):
        with patch("ailias.generation.unreal_session.process_identity",
                   return_value={**self.identity, "created": 456}):
            self.assertEqual(self.session.status()["status"], "stopped")
        atomic_json(self.session.state, {"session_id": "older", "status": "completed", "job_id": "stale"})
        with patch("ailias.generation.unreal_session.process_identity", return_value=self.identity):
            status = self.session.status()
            self.assertEqual(status["status"], "starting")
            self.assertNotIn("job_id", status)

    def test_wrong_project_and_wrong_live_editor_are_rejected(self):
        metadata = json.loads(self.session.metadata.read_text())
        metadata["project"] = str(self.root / "Other.uproject")
        atomic_json(self.session.metadata, metadata)
        with self.assertRaisesRegex(ValueError, "another project"):
            self.session.status()
        metadata["project"] = str(self.project)
        metadata["process"]["executable"] = "different.exe"
        atomic_json(self.session.metadata, metadata)
        with patch("ailias.generation.unreal_session.process_identity", return_value=metadata["process"]):
            with self.assertRaisesRegex(ValueError, "another editor"):
                self.session.status()

    def test_live_process_fingerprint_is_stable_and_includes_creation_time(self):
        first = process_identity(os.getpid())
        self.assertEqual(process_identity(os.getpid()), first)
        self.assertGreater(first["created"], 0)

    def test_stale_job_completion_cannot_complete_another_request(self):
        request = self.root / "request.json"
        receipt = self.root / "receipt.json"
        atomic_json(request, {"receipt": str(receipt)})
        status = {"status": "idle", "session_id": "owned", "pid": 42}
        with patch.object(self.session, "_start", return_value=(status, True)), \
                patch.object(self.session, "status", return_value={
                    **status, "status": "completed", "job_id": "older"}):
            with self.assertRaisesRegex(TimeoutError, "remains busy"):
                self.session.run(None, request, "session-probe", self.root, "done", .01)
        self.assertFalse(receipt.exists())

    def test_receipt_reuse_and_busy_editor_never_accept_a_new_job(self):
        request = self.root / "request.json"
        receipt = self.root / "receipt.json"
        atomic_json(request, {"receipt": str(receipt)})
        receipt.write_text("{}")
        status = {"status": "idle", "session_id": "owned", "pid": 42}
        with patch.object(self.session, "_start", return_value=(status, True)), \
                patch.object(self.session, "status", return_value=status):
            with self.assertRaisesRegex(ValueError, "existing Unreal job receipt"):
                self.session.run(None, request, "session-probe", self.root, "done", 1)
        with patch("ailias.generation.unreal_session.process_identity", return_value=self.identity):
            atomic_json(self.session.state, {"session_id": "owned", "status": "busy"})
            with self.assertRaisesRegex(ValueError, "busy"):
                self.session.assert_idle()
        self.assertFalse(self.session.inbox.exists())

    def test_submitted_but_not_yet_acknowledged_job_cannot_be_overwritten(self):
        atomic_json(self.session.inbox, {"session_id": "owned", "job_id": "pending"})
        with patch("ailias.generation.unreal_session.process_identity", return_value=self.identity):
            self.assertEqual(self.session.status()["status"], "pending")
            with self.assertRaisesRegex(ValueError, "busy"):
                self.session.assert_idle()
            with self.assertRaisesRegex(ValueError, "running a job"):
                self.session.stop()

    def test_persistent_jobs_reuse_the_owned_session_and_independent_verification_stays_cold(self):
        tools = NativeTools(self.root)
        tools.paths["unreal"] = self.editor
        settings = {"mode": "persistent", "startup_timeout_seconds": 180, "poll_seconds": .1}
        with patch("ailias.generation.unreal_session.load_settings", return_value=settings), \
                patch.object(UnrealSession, "run", return_value={"pid": 42}) as warm, \
                patch.object(tools, "run", return_value={"seconds": 1}) as cold, \
                patch("ailias.generation.unreal_session.process_identity", return_value=self.identity):
            for _ in range(2):
                result = execute_unreal(tools, self.project, "worker.py", "request.json", "edit",
                                        self.root, "REQUEST", "done")
                self.assertEqual(result["pid"], 42)
            execute_unreal(tools, self.project, "worker.py", "request.json", "verify", self.root,
                           "REQUEST", "done", independent=True)
        self.assertEqual(warm.call_count, 2)
        self.assertEqual(cold.call_count, 1)
        self.assertEqual(cold.call_args.kwargs["environment"], {"REQUEST": "request.json"})

    def test_explicit_oneshot_mode_does_not_start_a_persistent_editor(self):
        tools = NativeTools(self.root)
        tools.paths["unreal"] = self.editor
        settings = {"mode": "oneshot", "startup_timeout_seconds": 180, "poll_seconds": .1}
        with patch("ailias.generation.unreal_session.load_settings", return_value=settings), \
                patch.object(UnrealSession, "run") as warm, \
                patch.object(tools, "run", return_value={"seconds": 1}) as cold, \
                patch("ailias.generation.unreal_session.process_identity", return_value=self.identity):
            execute_unreal(tools, self.project, "worker.py", "request.json", "edit", self.root,
                           "REQUEST", "done")
        warm.assert_not_called()
        cold.assert_called_once()

    def test_native_session_failures_are_reported_as_tool_errors_without_automatic_retry(self):
        tools = NativeTools(self.root)
        tools.paths["unreal"] = self.editor
        with patch.object(UnrealSession, "run", side_effect=TimeoutError("still active")) as run:
            with self.assertRaisesRegex(ToolError, "still active"):
                execute_unreal(tools, self.project, "worker.py", "request.json", "edit", self.root,
                               "REQUEST", "done")
        run.assert_called_once()

    def test_session_status_does_not_initialize_reconstruction_or_probe_other_tools(self):
        from types import SimpleNamespace

        args = SimpleNamespace(generation_action="unreal-session", session_action="status",
                               project=self.project, editor=self.editor, wait=False)
        with patch("ailias.generation.cli.load_settings", side_effect=AssertionError), \
                patch("ailias.generation.cli.NativeTools", side_effect=AssertionError), \
                patch.object(UnrealSession, "status", return_value={"status": "idle"}):
            result, code = execute(args, self.root)
        self.assertEqual((result["status"], code), ("idle", 0))

    def test_invalid_wait_mode_is_rejected_before_starting_any_native_work(self):
        from types import SimpleNamespace

        args = SimpleNamespace(generation_action="unreal-session", session_action="probe",
                               project=self.project, editor=self.editor, wait=True)
        with patch.object(UnrealSession, "probe", side_effect=AssertionError) as probe:
            with self.assertRaisesRegex(ValueError, "only supported"):
                execute(args, self.root)
        probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()

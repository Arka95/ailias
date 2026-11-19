"""Exercise serialized asynchronous jobs using the real dispatcher and fake Unreal APIs."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ailias.generation.unreal_session_worker import Dispatcher
from ailias.storage import atomic_json


class UnrealDispatcherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = {"session_id": "owned", "project": "Project.uproject"}
        self.unreal = SimpleNamespace(
            SystemLibrary=SimpleNamespace(get_engine_version=lambda: "5.8.4-fixture"),
            EditorPythonScripting=SimpleNamespace(set_keep_python_script_alive=Mock()),
            unregister_slate_post_tick_callback=Mock(),
        )
        self.dispatcher = Dispatcher(self.root, self.metadata, self.unreal)

    def enqueue(self, job_id, worker=None):
        receipt = self.root / (job_id + "-receipt.json")
        envelope = {
            **self.metadata,
            "job_id": job_id,
            "operation": "session-probe",
            "worker": str(worker) if worker else None,
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest()
            if worker
            else None,
            "request": {"receipt": str(receipt)},
            "terminal_status": "session_probe_completed",
        }
        atomic_json(self.root / "job.json", envelope)
        return receipt

    def tick(self):
        self.dispatcher.last_poll = 0
        self.dispatcher.tick(0)
        return json.loads((self.root / "state.json").read_text())

    def test_two_distinct_jobs_complete_without_shutting_down_or_replaying_a_receipt(
        self,
    ):
        for job_id in ("first", "second"):
            receipt = self.enqueue(job_id)
            state = self.tick()
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["job_id"], job_id)
            self.assertEqual(
                json.loads(receipt.read_text())["status"], "session_probe_completed"
            )
            before = receipt.stat().st_mtime_ns
            self.tick()
            self.assertEqual(receipt.stat().st_mtime_ns, before)
        self.unreal.EditorPythonScripting.set_keep_python_script_alive.assert_not_called()

    def test_async_job_keeps_its_module_alive_and_does_not_admit_a_second_job_until_completion(
        self,
    ):
        first = self.enqueue("first", Path(__file__))
        started = []
        module = SimpleNamespace(
            main=lambda request, keep_alive: started.append((request, keep_alive))
        )
        with patch("ailias.generation.unreal_session_worker.exec", create=True):
            with patch(
                "ailias.generation.unreal_session_worker.importlib.util.module_from_spec",
                return_value=module,
            ):
                state = self.tick()
        self.assertEqual(state["status"], "busy")
        self.assertIs(self.dispatcher.module, module)
        self.assertTrue(started[0][1])
        second = self.enqueue("second")
        self.tick()
        self.assertFalse(second.exists())
        atomic_json(first, {"status": "session_probe_completed"})
        self.assertEqual(self.tick()["job_id"], "first")
        self.assertIsNone(self.dispatcher.module)
        self.assertEqual(self.tick()["job_id"], "second")
        self.assertTrue(second.is_file())

    def test_changed_worker_and_wrong_project_fail_without_running_a_job(self):
        receipt = self.enqueue("changed", Path(__file__))
        envelope = json.loads((self.root / "job.json").read_text())
        envelope["worker_sha256"] = "wrong"
        atomic_json(self.root / "job.json", envelope)
        state = self.tick()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(json.loads(receipt.read_text())["status"], "failed")
        self.enqueue("other-project")
        envelope = json.loads((self.root / "job.json").read_text())
        envelope["project"] = "Other.uproject"
        atomic_json(self.root / "job.json", envelope)
        self.assertEqual(self.tick()["error_type"], "ProjectMismatch")

    def test_unwritable_receipt_fails_the_probe_without_wedging_the_next_job(self):
        receipt = self.enqueue("unwritable")
        from ailias.generation.unreal_session_worker import write_json

        def persist(path, value):
            if path == receipt:
                raise PermissionError("fixture receipt cannot be written")
            return write_json(path, value)

        with patch(
            "ailias.generation.unreal_session_worker.write_json", side_effect=persist
        ):
            state = self.tick()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error_type"], "PermissionError")
        self.assertEqual(state["receipt_error_type"], "PermissionError")
        self.assertIsNone(self.dispatcher.active)
        self.assertFalse(receipt.exists())
        next_receipt = self.enqueue("next")
        self.assertEqual(self.tick()["status"], "completed")
        self.assertTrue(next_receipt.exists())

    def test_native_slate_pumping_cannot_reenter_dispatch_or_steal_build_exceptions(
        self,
    ):
        callbacks = [self.dispatcher.tick]
        self.dispatcher.callback = callbacks[0]
        self.unreal.unregister_slate_post_tick_callback = callbacks.remove

        def register(callback):
            callbacks.append(callback)
            return callback

        self.unreal.register_slate_post_tick_callback = register
        receipt = self.enqueue("native-build", Path(__file__))

        def build(request, keep_alive):
            if callbacks:
                raise SystemError(
                    "Reentrant Python callback consumed the native build exception"
                )
            raise RuntimeError("Original native build error")

        with (
            patch("ailias.generation.unreal_session_worker.exec", create=True),
            patch(
                "ailias.generation.unreal_session_worker.importlib.util.module_from_spec",
                return_value=SimpleNamespace(main=build),
            ),
        ):
            self.assertEqual(self.tick()["status"], "failed")
        self.assertEqual(json.loads(receipt.read_text())["error_type"], "RuntimeError")
        self.assertEqual(callbacks, [self.dispatcher.tick])
        self.assertIsNotNone(self.dispatcher.callback)

    def test_unknown_completion_status_does_not_release_the_editor(self):
        receipt = self.enqueue("async", Path(__file__))
        envelope = json.loads((self.root / "job.json").read_text())
        self.dispatcher.active = envelope
        atomic_json(receipt, {"status": "rendering"})
        self.dispatcher.inspect_completion()
        self.assertIsNotNone(self.dispatcher.active)
        atomic_json(receipt, {"status": "failed"})
        self.dispatcher.inspect_completion()
        self.assertIsNone(self.dispatcher.active)
        self.assertEqual(
            json.loads((self.root / "state.json").read_text())["status"], "failed"
        )

    def test_old_async_receipt_cannot_finish_the_next_job(self):
        first = self.enqueue("first")
        self.assertEqual(self.tick()["status"], "completed")
        second = self.enqueue("second", Path(__file__))
        envelope = json.loads((self.root / "job.json").read_text())
        self.dispatcher.active = envelope
        self.dispatcher.publish("busy")
        atomic_json(first, {"status": "session_probe_completed", "late_callback": True})
        self.tick()
        self.assertEqual(self.dispatcher.active["job_id"], "second")
        self.assertFalse(second.exists())
        atomic_json(second, {"status": "session_probe_completed"})
        self.assertEqual(self.tick()["job_id"], "second")

    def test_render_console_variables_are_restored_before_another_job_is_admitted(self):
        calls = []
        self.unreal.UnrealEditorSubsystem = object()
        self.unreal.get_editor_subsystem = lambda subsystem: SimpleNamespace(
            get_editor_world=lambda: "world"
        )
        self.unreal.SystemLibrary.get_console_variable_float_value = lambda name: 1
        self.unreal.SystemLibrary.execute_console_command = lambda world, command: (
            calls.append((world, command))
        )
        receipt = self.enqueue("render")
        envelope = json.loads((self.root / "job.json").read_text())
        envelope["console_commands"] = "r.DefaultFeature.AutoExposure 0"
        atomic_json(self.root / "job.json", envelope)
        self.assertEqual(self.tick()["status"], "completed")
        self.assertEqual(
            calls,
            [
                ("world", "r.DefaultFeature.AutoExposure 0"),
                ("world", "r.DefaultFeature.AutoExposure 1"),
            ],
        )
        self.assertTrue(receipt.is_file())


if __name__ == "__main__":
    unittest.main()

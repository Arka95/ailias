"""Native failure receipts survive process errors; no external processes are launched."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from ailias.facefitting.unreal.worker_process import execute_unreal_worker
from ailias.generation.tools import ToolError
from ailias.storage import atomic_json


class NativeJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tools = Mock()
        self.report = {}

    def launch(self, receipt=None, failure=None):
        def execute(*args, **kwargs):
            request = json.loads(Path(kwargs["environment"]["AILIAS_FRESH_VIDEO_REQUEST"]).read_text())
            if receipt is not None:
                atomic_json(Path(request["receipt"]), receipt)
            if failure is not None:
                raise failure
            return {"seconds": 1}
        self.tools.run.side_effect = execute
        return execute_unreal_worker(self.tools, Path("Project.uproject"), Path("worker.py"),
                              self.root, "verify", {}, self.report)

    def test_nonzero_process_exit_preserves_native_diagnostics_and_exception_chain(self):
        failure = ToolError("exit_1; inspect log")
        with self.assertRaisesRegex(RuntimeError, "precise geometry failure") as caught:
            self.launch({"status": "failed", "error": "precise geometry failure",
                         "maximum_distance_cm": .25}, failure)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(self.report["verify"]["maximum_distance_cm"], .25)

    def test_missing_receipt_after_process_failure_keeps_both_failure_contexts(self):
        with self.assertRaisesRegex(RuntimeError, "without a readable receipt.*exit_1"):
            self.launch(failure=ToolError("exit_1"))

    def test_successful_process_does_not_make_a_failed_native_operation_successful(self):
        with self.assertRaisesRegex(RuntimeError, "rejected.*save failed"):
            self.launch({"status": "failed", "error": "save failed"})

    def test_zero_exit_with_an_incomplete_receipt_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, "did not complete.*verifying"):
            self.launch({"status": "verifying"})

    def test_success_requires_a_json_object_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "no valid receipt"):
            self.launch(["not", "an", "object"])

    def test_valid_receipt_is_returned_and_runtime_is_recorded(self):
        self.assertEqual(self.launch({"status": "fresh_character_reloaded"})["status"],
                         "fresh_character_reloaded")
        self.assertEqual(self.report["verify_runtime"]["seconds"], 1)
        arguments = self.tools.run.call_args.args[2]
        self.assertIn("-d3d12", arguments)
        self.assertIn("-ExecutePythonScript=worker.py", arguments)
        self.assertNotIn("-NullRHI", arguments)
        self.assertNotIn("-run=pythonscript", arguments)

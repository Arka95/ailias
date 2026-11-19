"""Execution chains stop on failure and reliably release only owned resources."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ailias.common.execution import ExecutionChain, PipelineStep, StageJournal


class ExecutionTests(unittest.TestCase):
    def test_chain_preserves_order_and_cleans_each_step_once(self):
        events = []
        ExecutionChain(
            [
                PipelineStep(
                    "first",
                    lambda: events.append("first"),
                    lambda: events.append("close_first"),
                ),
                PipelineStep("second", lambda: events.append("second")),
            ]
        ).run()
        self.assertEqual(events, ["first", "close_first", "second"])

    def test_failure_stops_chain_but_still_cleans_failed_step(self):
        events = []

        def fail():
            raise ValueError("failed")

        with self.assertRaises(ValueError):
            ExecutionChain(
                [
                    PipelineStep("first", fail, lambda: events.append("close")),
                    PipelineStep("never", lambda: events.append("never")),
                ]
            ).run()
        self.assertEqual(events, ["close"])

    def test_cleanup_error_does_not_replace_original_failure(self):
        def fail():
            raise ValueError("original")

        def cleanup():
            raise OSError("cleanup")

        with self.assertRaisesRegex(ValueError, "original") as caught:
            ExecutionChain([PipelineStep("first", fail, cleanup)]).run()
        self.assertIn("OSError", caught.exception.__notes__[0])

    def test_journal_persists_failed_stage_and_nested_cause(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            report = {}

            def fail():
                try:
                    raise OSError("input")
                except OSError as cause:
                    raise RuntimeError("stage") from cause

            with self.assertRaises(RuntimeError):
                StageJournal(report, manifest).execute("load", fail)
            stage = json.loads(manifest.read_text())["stages"]["load"]
            self.assertEqual(stage["status"], "failed")
            self.assertEqual(stage["cause"], "input")
            self.assertIn("seconds", stage)

    def test_interrupt_also_cleans_owned_resources(self):
        events = []

        def interrupt():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            ExecutionChain(
                [PipelineStep("first", interrupt, lambda: events.append("closed"))]
            ).run()
        self.assertEqual(events, ["closed"])

    def test_progress_is_safe_and_observer_failure_never_changes_the_outcome(self):
        events = []

        def observer(event):
            events.append(event)
            raise ValueError("observer failed")

        journal = StageJournal({}, on_progress=observer)
        self.assertEqual(journal.execute("success", lambda: 42), 42)

        def fail():
            raise RuntimeError("private diagnostic detail")

        with self.assertRaisesRegex(RuntimeError, "private diagnostic detail"):
            journal.execute("failure", fail)
        self.assertEqual(events[-1]["error_type"], "RuntimeError")
        self.assertNotIn("private diagnostic detail", json.dumps(events))
        self.assertNotIn("error", events[-1])
        self.assertNotIn("cause", events[-1])

    def test_asynchronous_stage_measures_until_completion_not_until_launch_returns(
        self,
    ):
        now = [10.0]
        with patch(
            "ailias.common.execution.time.monotonic", side_effect=lambda: now[0]
        ):
            report = {}
            journal = StageJournal(report)
            journal.begin("render")
            now[0] = 12.0
            self.assertEqual(report["stages"]["render"]["status"], "running")
            self.assertNotIn("seconds", report["stages"]["render"])
            now[0] = 25.0
            journal.finish("render")
        self.assertEqual(report["stages"]["render"]["seconds"], 15.0)

    def test_failed_manifest_write_does_not_replace_the_original_stage_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = StageJournal({}, Path(temporary) / "manifest.json")

            def fail():
                raise RuntimeError("original")

            with patch(
                "ailias.storage.atomic_json", side_effect=[None, PermissionError()]
            ):
                with self.assertRaisesRegex(RuntimeError, "original") as caught:
                    journal.execute("fixture", fail)
            self.assertIn("PermissionError", caught.exception.__notes__[0])

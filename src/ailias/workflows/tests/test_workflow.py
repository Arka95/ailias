from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

from ailias.imageprocessing.face_features import FaceFeatures, FaceQuality
from ailias.workflows import Action, IAction, IWorkflow, Workflow


class RecordingAction(Action):
    """Provide deterministic action output, exports and owned scratch without external integrations.
    Record lifecycle calls so tests can observe ordering, handoff and cleanup after partial failures.
    """

    def __init__(self, name, events, operation=lambda value: value):
        """Associate this fixture with a readable operation name and a pure output transformation.
        Keep failures configurable without giving the action any run identifier.
        """
        self.name, self.events, self.operation = name, events, operation
        self.execution_error = None
        self.cleanup_error = None

    def execute(self, context):
        """Record the previous output and create one action-owned temporary file before transforming it.
        Allow an injected failure after allocation to exercise cleanup of partially executed operations.
        """
        self.events.append(("execute", self.name, context.current))
        (context.scratch_directory / (self.name + ".tmp")).write_text("owned scratch", encoding="utf-8")
        if self.execution_error is not None:
            raise self.execution_error
        context.data[self.name] = context.current
        return self.operation(context.current)

    def save(self, context):
        """Export one durable fixture asset using the latest shared context, not a replayed operation.
        Keep it outside scratch so action cleanup cannot remove saved results.
        """
        self.events.append(("save", self.name, context.current))
        (context.artifacts_directory / (self.name + ".txt")).write_text("saved output", encoding="utf-8")

    def clean(self, context):
        """Remove only this action's scratch file and retain any durable export.
        Simulate cleanup errors before deletion so a subsequent cleanup can retry that action alone.
        """
        self.events.append(("clean", self.name))
        if self.cleanup_error is not None:
            raise self.cleanup_error
        (context.scratch_directory / (self.name + ".tmp")).unlink(missing_ok=True)


@dataclass
class SavedReference:
    """Exercise generic dataclass/path serialization without depending on a model-specific adapter.
    Retain only a caller-selected file reference and its numeric observation count.
    """

    path: Path
    observations: int


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        """Isolate all workflow directories and source fixtures below this test's temporary root.
        Use only ordinary filesystem operations and fake actions, never models or native tools.
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / "runs"
        self.events = []

    def test_action_chain_uses_previous_output_and_shared_context_without_implicit_save_or_cleanup(self):
        """Pass successful outputs through the same context in order and expose the final result.
        Keep persistence and cleanup separate until their explicit lifecycle methods are called.
        """
        first = RecordingAction("increment", self.events, lambda value: value + 1)
        second = RecordingAction("double", self.events, lambda value: value * 2)
        workflow = Workflow([first, second], directory=self.runs, initial=3)
        self.assertFalse(self.runs.exists())
        self.assertEqual(workflow.execute(), 8)
        self.assertEqual(workflow.context.current, 8)
        self.assertEqual(workflow.context.data, {"increment": 3, "double": 4})
        self.assertEqual(self.events, [("execute", "increment", 3), ("execute", "double", 4)])
        self.assertFalse((workflow.context.run_directory / "context.json").exists())
        self.assertTrue((workflow.context.scratch_directory / "increment.tmp").exists())
        snapshot = workflow.save()
        self.assertEqual(json.loads(snapshot.read_text())["current"], 8)
        self.assertEqual(self.events[2:], [("save", "increment", 8), ("save", "double", 8)])
        workflow.clean()
        self.assertEqual(self.events[-2:], [("clean", "double"), ("clean", "increment")])

    def test_save_serializes_current_face_features_and_named_context_without_domain_specific_workflow_code(self):
        """Save real FaceFeatures/FaceQuality objects through their to_dict contracts without inference.
        Convert nested dataclasses, file references and the workflow UUID into ordinary JSON values.
        """
        face = FaceFeatures(orientation={"pitch_deg": 35, "yaw_deg": -75, "roll_deg": 0, "calibrated": False},
                            quality=FaceQuality(supported_landmarks=450, reasons=["face_too_small"]))
        workflow = Workflow([], directory=self.runs, initial=face)
        workflow.context.data["source"] = SavedReference(self.root / "original.png", 478)
        workflow.context.data["run_reference"] = workflow.id
        snapshot = workflow.save()
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        self.assertEqual(payload["current"], face.to_dict())
        self.assertEqual(payload["run_id"], str(workflow.id))
        self.assertEqual(payload["data"]["source"], {"path": str(self.root / "original.png"), "observations": 478})
        self.assertEqual(payload["data"]["run_reference"], str(workflow.id))
        self.assertIs(workflow.context.current, face)
        changed = FaceFeatures(quality=FaceQuality(max_expression_score=.9, reasons=["non_neutral_expression"]))
        workflow.context.current = changed
        self.assertEqual(workflow.save(), snapshot)
        self.assertEqual(json.loads(snapshot.read_text())["current"], changed.to_dict())

    def test_workflow_guids_and_directories_isolate_runs_under_the_same_output_root(self):
        """Allocate one GUID/directory per workflow instance and retain independent saved contexts.
        Keep run identity stable across execution and saving rather than creating action GUIDs.
        """
        first = Workflow([], directory=self.runs, initial="first")
        second = Workflow([], directory=self.runs, initial="second")
        self.assertIsInstance(first.id, UUID)
        self.assertNotEqual(first.id, second.id)
        first.execute()
        first_file, second_file = first.save(), second.save()
        self.assertEqual(first_file.parent.name, str(first.id))
        self.assertNotEqual(first_file.parent, second_file.parent)
        self.assertEqual(json.loads(first_file.read_text())["current"], "first")
        self.assertEqual(json.loads(second_file.read_text())["current"], "second")

    def test_direct_context_snapshot_and_workflow_execution_share_the_same_run_reservation(self):
        """Allow saving shared context before execution without mistaking its directory for another run.
        Preserve the run identity and let later workflow snapshots update the same context file.
        """
        action = RecordingAction("increment", self.events, lambda value: value + 1)
        workflow = Workflow([action], directory=self.runs, initial=3)
        snapshot = workflow.context.save()
        self.assertEqual(json.loads(snapshot.read_text())["current"], 3)
        self.assertEqual(workflow.execute(), 4)
        self.assertEqual(workflow.save(), snapshot)
        self.assertEqual(json.loads(snapshot.read_text())["current"], 4)
        workflow.clean()

    def test_execution_failure_stops_later_actions_and_allows_saving_last_successful_context(self):
        """Retain the preceding successful output when an atomic action raises after allocating scratch.
        Export only completed actions and clean both completed and failed actions without touching originals.
        """
        original = self.root / "original.txt"
        original.write_bytes(b"preserve source")
        first = RecordingAction("first", self.events, lambda value: value + 1)
        failed = RecordingAction("failed", self.events)
        failed.execution_error = ValueError("fixture failure")
        never = RecordingAction("never", self.events)
        workflow = Workflow([first, failed, never], directory=self.runs, initial=1)
        workflow.context.data["source"] = original
        with self.assertRaisesRegex(ValueError, "fixture failure"):
            workflow.execute()
        self.assertEqual(workflow.context.current, 2)
        snapshot = workflow.save()
        before = snapshot.read_bytes()
        unrelated = workflow.context.scratch_directory / "not-owned-by-actions.tmp"
        unrelated.write_bytes(b"preserve unregistered file")
        workflow.clean()
        workflow.clean()
        self.assertEqual([event for event in self.events if event[0] == "clean"],
                         [("clean", "failed"), ("clean", "first")])
        self.assertNotIn(("execute", "never", 2), self.events)
        self.assertEqual(json.loads(before)["current"], 2)
        self.assertEqual(snapshot.read_bytes(), before)
        self.assertTrue((workflow.context.artifacts_directory / "first.txt").is_file())
        self.assertEqual(original.read_bytes(), b"preserve source")
        self.assertEqual(unrelated.read_bytes(), b"preserve unregistered file")
        self.assertFalse((workflow.context.scratch_directory / "failed.tmp").exists())

    def test_cleanup_continues_after_failure_and_retries_only_the_failed_hook(self):
        """Release other entered actions when one cleanup raises and report the cleanup failures together.
        Preserve the failed hook for retry without invoking successful cleanup hooks twice.
        """
        first = RecordingAction("first", self.events)
        second = RecordingAction("second", self.events)
        second.cleanup_error = OSError("fixture cleanup")
        workflow = Workflow([first, second], directory=self.runs)
        workflow.execute()
        with self.assertRaises(ExceptionGroup) as caught:
            workflow.clean()
        self.assertIsInstance(caught.exception.exceptions[0], OSError)
        self.assertEqual(self.events[-2:], [("clean", "second"), ("clean", "first")])
        second.cleanup_error = None
        workflow.clean()
        workflow.clean()
        self.assertEqual([event for event in self.events if event[0] == "clean"],
                         [("clean", "second"), ("clean", "first"), ("clean", "second")])

    def test_cleanup_failure_does_not_replace_an_active_execution_failure(self):
        """Preserve the original exception when explicit cleanup runs in the caller's finally block.
        Attach cleanup error types to that exception rather than hiding or replacing the primary failure.
        """
        action = RecordingAction("failed", self.events)
        action.execution_error = ValueError("primary fixture")
        action.cleanup_error = OSError("cleanup fixture")
        workflow = Workflow([action], directory=self.runs)
        with self.assertRaisesRegex(ValueError, "primary fixture") as caught:
            try:
                workflow.execute()
            finally:
                workflow.clean()
        self.assertTrue(any("OSError" in note for note in caught.exception.__notes__))
        action.cleanup_error = None
        workflow.clean()

    def test_interruption_still_registers_the_entered_action_for_explicit_cleanup(self):
        """Preserve an execution interrupt while retaining partial action resources for explicit cleanup.
        Do not enter subsequent actions after a keyboard interruption.
        """
        action = RecordingAction("interrupted", self.events)
        action.execution_error = KeyboardInterrupt()
        workflow = Workflow([action, RecordingAction("never", self.events)], directory=self.runs)
        with self.assertRaises(KeyboardInterrupt):
            try:
                workflow.execute()
            finally:
                workflow.clean()
        self.assertEqual(self.events, [("execute", "interrupted", None), ("clean", "interrupted")])

    def test_invalid_context_cannot_replace_the_last_valid_snapshot(self):
        """Reject opaque runtime objects, nonfinite values and reference cycles during context serialization.
        Leave the previous context JSON unchanged instead of silently stringifying or partially writing data.
        """
        workflow = Workflow([], directory=self.runs, initial={"valid": True})
        snapshot = workflow.save()
        previous = snapshot.read_bytes()
        cyclic = []
        cyclic.append(cyclic)
        for value, error_type in ((object(), TypeError), (float("nan"), ValueError), (cyclic, ValueError)):
            with self.subTest(error_type=error_type):
                workflow.context.current = value
                with self.assertRaises(error_type):
                    workflow.save()
                self.assertEqual(snapshot.read_bytes(), previous)

    def test_dataclass_serialization_rejects_runtime_resources_without_copying_them(self):
        """Inspect ordinary dataclass fields without deep-copying opaque resource handles during saving.
        Require an explicit adapter for runtime objects rather than accidentally cloning their state.
        """
        class RuntimeResource:
            def __deepcopy__(self, memo):
                """Reject unexpected resource duplication so this fixture detects implicit cloning.
                Never emulate a real model, device or external handle in the unit test.
                """
                raise AssertionError("A context snapshot must not copy runtime resources")

        workflow = Workflow([], directory=self.runs)
        workflow.context.data["resource"] = SavedReference(RuntimeResource(), 1)
        with self.assertRaisesRegex(TypeError, "RuntimeResource"):
            workflow.save()
        self.assertFalse((workflow.context.run_directory / "context.json").exists())

    def test_save_hook_failure_does_not_publish_a_new_context_snapshot(self):
        """Leave the previous snapshot intact when an action's explicit artifact-export hook fails.
        Never rerun execution to retry saving the latest in-memory context.
        """
        action = RecordingAction("action", self.events, lambda value: value + 1)
        workflow = Workflow([action], directory=self.runs, initial=2)
        snapshot = workflow.save()
        previous = snapshot.read_bytes()
        workflow.execute()
        with patch.object(action, "save", side_effect=OSError("fixture export")), self.assertRaises(OSError):
            workflow.save()
        self.assertEqual(snapshot.read_bytes(), previous)
        self.assertEqual(workflow.context.current, 3)
        workflow.save()
        self.assertEqual(json.loads(snapshot.read_text())["current"], 3)
        self.assertEqual(len([event for event in self.events if event[0] == "execute"]), 1)
        workflow.clean()

    def test_existing_run_directory_is_not_adopted_or_overwritten(self):
        """Refuse an already existing run directory even if a generated GUID unexpectedly collides.
        Preserve its saved context and keep actions from executing inside another run.
        """
        first = Workflow([], directory=self.runs, initial="existing")
        snapshot = first.save()
        previous = snapshot.read_bytes()
        with patch("ailias.workflows.workflow.uuid4", return_value=first.id):
            collision = Workflow([RecordingAction("never", self.events)], directory=self.runs)
        with self.assertRaises(FileExistsError):
            collision.execute()
        self.assertEqual(snapshot.read_bytes(), previous)
        self.assertEqual(self.events, [])

    def test_execute_cannot_replay_a_run_and_cleaned_workflows_cannot_export_again(self):
        """Require a new workflow instance for another execution rather than reusing a run identity.
        Prevent saving through released action resources after explicit cleanup.
        """
        action = RecordingAction("action", self.events)
        workflow = Workflow([action], directory=self.runs, initial=3)
        workflow.execute()
        with self.assertRaises(RuntimeError):
            workflow.execute()
        snapshot = workflow.save()
        workflow.clean()
        with self.assertRaises(RuntimeError):
            workflow.execute()
        with self.assertRaises(RuntimeError):
            workflow.save()
        self.assertTrue(snapshot.is_file())

    def test_resource_free_actions_need_only_execute_and_interfaces_require_complete_implementations(self):
        """Allow pure atomic transformations to inherit default save/clean hooks without resource boilerplate.
        Enforce the public abstract contracts and reject unrelated objects before creating run directories.
        """
        class Increment(Action):
            def execute(self, context):
                """Return an incremented scalar without allocating separate resources.
                Use the containing workflow to persist the resulting shared context.
                """
                return context.current + 1

        for contract in (IWorkflow, IAction, Action):
            with self.subTest(contract=contract), self.assertRaises(TypeError):
                contract()
        with self.assertRaises(TypeError):
            Workflow([object()], directory=self.runs)
        self.assertFalse(self.runs.exists())
        workflow = Workflow([Increment()], directory=self.runs, initial=1)
        self.assertEqual(workflow.execute(), 2)
        snapshot = workflow.save()
        workflow.clean()
        self.assertEqual(json.loads(snapshot.read_text())["current"], 2)

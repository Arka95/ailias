"""Single-run action chains with explicit context snapshots and reverse-order cleanup."""

from collections.abc import Iterable
from pathlib import Path
import sys
from typing import Any
from uuid import UUID, uuid4

from ailias.workflows.context import WorkflowContext
from ailias.workflows.contracts import IAction, IWorkflow


class Workflow(IWorkflow):
    """Execute an ordered chain of atomic actions under one run GUID and shared context.
    Save and cleanup are explicit; existing jobs are not automatically migrated to this runtime.
    """

    def __init__(self, actions: Iterable[IAction], *, directory: str | Path, initial: Any = None):
        """Declare a single-run action chain with an explicit output root and initial context value.
        Allocate its GUID immediately but create directories only when execution or saving begins.
        """
        self._actions = tuple(actions)
        if not all(isinstance(action, IAction) for action in self._actions):
            raise TypeError("Workflow actions must implement IAction")
        run_id = uuid4()
        self._context = WorkflowContext(run_id, Path(directory).resolve() / str(run_id), current=initial)
        self._started = False
        self._cleaned = False
        self._completed: list[IAction] = []
        self._pending_cleanup: list[IAction] = []

    @property
    def id(self) -> UUID:
        """Return the GUID identifying this workflow run rather than any individual action.
        Construct a new Workflow instance to obtain a separate run and identity.
        """
        return self.context.run_id

    @property
    def context(self) -> WorkflowContext:
        """Expose this run's latest output, shared values and conventional artifact/scratch paths.
        Keep the context instance stable so consecutive actions observe the same shared state.
        """
        return self._context

    def execute(self) -> Any:
        """Run actions in order, updating context.current only when each operation returns successfully.
        Stop on the original failure and retain entered actions for explicit reverse-order cleanup.
        """
        if self._started or self._cleaned:
            raise RuntimeError("Workflow runs execute once; create a new Workflow for another run")
        self.context.prepare_directories()
        self._started = True
        for action in self._actions:
            self._pending_cleanup.append(action)
            result = action.execute(self.context)
            self.context.current = result
            self._completed.append(action)
        return self.context.current

    def save(self) -> Path:
        """Run completed actions' export hooks, then atomically snapshot the current shared context.
        Permit saving initial or partially completed state without executing or replaying any action.
        """
        if self._cleaned:
            raise RuntimeError("Save the workflow context before cleaning its resources")
        self.context.prepare_directories()
        for action in self._completed:
            action.save(self.context)
        return self.context.save()

    def clean(self) -> None:
        """Clean entered actions in reverse order while preserving context JSON and durable outputs.
        Continue after cleanup failures, retry only failed hooks, and never mask an active primary exception.
        """
        self._cleaned = True
        primary = sys.exception()
        failures, pending = [], []
        for action in reversed(self._pending_cleanup):
            try:
                action.clean(self.context)
            except BaseException as error:
                failures.append(error)
                pending.append(action)
        self._pending_cleanup = list(reversed(pending))
        if failures:
            if primary is not None:
                for error in failures:
                    primary.add_note(f"Action cleanup also failed: {type(error).__name__}")
            else:
                raise BaseExceptionGroup("Workflow action cleanup failed", failures)

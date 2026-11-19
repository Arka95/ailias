"""Abstract workflow/action lifecycle contracts and defaults for resource-free actions."""

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any
from uuid import UUID

from ailias.workflows.context import WorkflowContext


class IWorkflow(metaclass=ABCMeta):
    """Require explicit execution, context persistence and owned-resource cleanup on workflow objects.
    Implementations represent workflow runs, not identities for individual actions.
    """

    @property
    @abstractmethod
    def id(self) -> UUID:
        """Expose the GUID associated with this workflow run rather than with any atomic action.
        Keep the run identity stable across execution, saving and cleanup.
        """

    @abstractmethod
    def execute(self) -> Any:
        """Execute the workflow's ordered action chain and return its latest output.
        Stop on failure without implicitly saving state or cleaning resources.
        """

    @abstractmethod
    def save(self) -> Path:
        """Persist the workflow's current shared context in its run directory.
        Save explicit data representations rather than arbitrary live runtime objects.
        """

    @abstractmethod
    def clean(self) -> None:
        """Release only this workflow's owned resources and temporary action outputs.
        Preserve original inputs, saved context and durable artifacts.
        """


class IAction(metaclass=ABCMeta):
    """Define one atomic operation with explicit persistence and cleanup hooks but no action GUID.
    Share inputs and output references through the containing workflow's context.
    """

    @abstractmethod
    def execute(self, context: WorkflowContext) -> Any:
        """Read the preceding output from context.current and return this operation's output.
        Use context.data or conventional run directories for named/intermediate evidence.
        """

    @abstractmethod
    def save(self, context: WorkflowContext) -> None:
        """Export action-owned durable assets when the containing workflow is explicitly saved.
        Use current shared state or owned results and leave JSON context persistence to the workflow.
        """

    @abstractmethod
    def clean(self, context: WorkflowContext) -> None:
        """Release owned handles and temporary files without removing saved outputs or source inputs.
        Support cleanup after partial execution, including a failed execute call.
        """


class Action(IAction):
    """Provide optional no-op save/cleanup hooks for atomic operations that own no external resources.
    Concrete subclasses still implement execute(); resource-owning actions override the relevant hooks.
    """

    def save(self, context: WorkflowContext) -> None:
        """Leave persistence to the workflow's JSON snapshot when no separate assets need exporting.
        Override this hook to save action-specific binary outputs and add their references to context.
        """

    def clean(self, context: WorkflowContext) -> None:
        """Perform no cleanup for a resource-free action without clearing retained context data.
        Override this hook when the operation owns handles or temporary files.
        """

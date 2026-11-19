"""Shared in-memory action state and explicit per-run JSON snapshots."""

from pathlib import Path
from typing import Any
from uuid import UUID

from ailias.common.json_values import to_json_value
from ailias.storage import atomic_json


class WorkflowContext:
    """Carry the latest action output and named shared state within one workflow run.
    Keep run identity/directory read-only while allowing actions to add domain-specific context values.
    """

    def __init__(self, run_id: UUID, run_directory: Path, *, current: Any = None):
        """Associate initial state with the workflow-owned GUID and directory without writing files.
        Give each context its own named-value dictionary rather than sharing mutable defaults.
        """
        self._run_id = run_id
        self._run_directory = run_directory
        self._directory_created = False
        self.current = current
        self.data: dict[str, Any] = {}

    @property
    def run_id(self) -> UUID:
        """Expose the containing workflow's run GUID without assigning identities to actions.
        Keep the same identity throughout execution, snapshots and cleanup.
        """
        return self._run_id

    @property
    def run_directory(self) -> Path:
        """Return the directory reserved for this run's snapshots and action-owned artifacts.
        Accessing the path does not create it or alter any external input.
        """
        return self._run_directory

    @property
    def artifacts_directory(self) -> Path:
        """Return the conventional location for persisted action outputs within this run.
        Keep durable artifacts separate from scratch that action cleanup may release.
        """
        return self.run_directory / "artifacts"

    @property
    def scratch_directory(self) -> Path:
        """Return the conventional location for temporary action-owned files within this run.
        Individual actions remain responsible for deleting only the scratch they created.
        """
        return self.run_directory / "scratch"

    def to_dict(self) -> dict:
        """Capture current and named context values using their explicit JSON-compatible representations.
        Include run identity without serializing action instances, resource handles or image bytes implicitly.
        """
        return to_json_value({
            "schema_version": 1, "run_id": self.run_id,
            "current": self.current, "data": self.data,
        })

    def prepare_directories(self) -> None:
        """Reserve this run directory exclusively and prepare conventional artifact/scratch paths.
        Share reservation state across workflow execution and direct context snapshots without adopting another run.
        """
        if not self._directory_created:
            self.run_directory.mkdir(parents=True, exist_ok=False)
            self._directory_created = True
        self.artifacts_directory.mkdir(exist_ok=True)
        self.scratch_directory.mkdir(exist_ok=True)

    def save(self) -> Path:
        """Atomically save the current context as context.json in this run's directory.
        Validate serialization before writing so an unsupported value cannot replace the last valid snapshot.
        """
        payload = self.to_dict()
        self.prepare_directories()
        destination = self.run_directory / "context.json"
        atomic_json(destination, payload)
        return destination

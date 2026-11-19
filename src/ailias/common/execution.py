"""Ordered execution with explicit resource ownership and durable stage results."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sys
import time
from typing import TypeVar

T = TypeVar("T")


def emit_stage_progress(event: dict) -> None:
    """Write bounded operation metadata to stderr, never diagnostic payloads."""
    print(json.dumps(event, allow_nan=False), file=sys.stderr, flush=True)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    execute: Callable[[], object]
    cleanup: Callable[[], None] | None = None


def execute_owned_step(
    operation: Callable[[], T], cleanup: Callable[[], None] | None = None
) -> T:
    failure = None
    try:
        return operation()
    except BaseException as error:
        failure = error
        raise
    finally:
        if cleanup is not None:
            try:
                cleanup()
            except Exception as error:
                if failure is None:
                    raise
                failure.add_note(f"Step cleanup also failed: {type(error).__name__}")


class ExecutionChain:
    """Run declared steps in order, stopping immediately after any failure."""

    def __init__(self, steps: Iterable[PipelineStep]):
        self.steps = tuple(steps)
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("Execution step names must be unique")

    def run(self, journal=None) -> None:
        for step in self.steps:
            if journal is None:
                execute_owned_step(step.execute, step.cleanup)
            else:
                journal.execute(step.name, step.execute, step.cleanup)


class StageJournal:
    """Record stage boundaries without owning domain logic or shared runtimes."""

    def __init__(
        self,
        report: dict,
        manifest=None,
        on_progress: Callable[[dict], None] | None = None,
    ):
        self.report, self.manifest, self.on_progress = report, manifest, on_progress
        self.started = time.monotonic()
        self._running = {}

    @property
    def seconds(self) -> float:
        return round(time.monotonic() - self.started, 6)

    @property
    def running_stages(self) -> tuple[str, ...]:
        return tuple(self._running)

    def persist(self, failure=None):
        if self.manifest is None:
            return
        from ailias.storage import atomic_json

        try:
            atomic_json(self.manifest, self.report)
        except Exception as error:
            if failure is None:
                raise
            failure.add_note(
                f"Stage journal persistence also failed: {type(error).__name__}"
            )

    def begin(self, name: str) -> None:
        if name in self._running:
            raise ValueError(f"Stage is already running: {name}")
        parent = next(reversed(self._running), None)
        self._running[name] = time.monotonic()
        self.report["current_stage"] = name
        self.report.setdefault("stages", {})[name] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "parent": parent,
        }
        self.persist()

    def finish(self, name: str, failure: BaseException | None = None) -> None:
        started = self._running.pop(name)
        entry = self.report["stages"][name]
        entry["status"] = "passed"
        if failure is not None:
            entry.update(
                status="failed" if isinstance(failure, Exception) else "interrupted",
                error_type=type(failure).__name__,
                error=str(failure),
                cause=str(failure.__cause__) if failure.__cause__ is not None else None,
            )
            self.report.setdefault("failed_stage", name)
        entry.update(
            seconds=round(time.monotonic() - started, 6),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        self.persist(failure)
        if self.on_progress is not None:
            event = {
                key: value
                for key, value in entry.items()
                if key
                in {
                    "status",
                    "started_at",
                    "finished_at",
                    "seconds",
                    "error_type",
                    "parent",
                }
            }
            try:
                self.on_progress({"stage": name, **event})
            except Exception:
                pass

    def execute(
        self,
        name: str,
        operation: Callable[[], T],
        cleanup: Callable[[], None] | None = None,
    ) -> T:
        self.begin(name)
        failure = None
        try:
            return execute_owned_step(operation, cleanup)
        except BaseException as error:
            failure = error
            raise
        finally:
            self.finish(name, failure)

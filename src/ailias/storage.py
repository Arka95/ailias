import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("intake", "reconstruction", "cleanup", "metahuman", "baseline")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class Store:
    def __init__(self, data_root: Path, session: str):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", session):
            raise ValueError("Session must be 1-64 letters, digits, underscores or hyphens")
        base = data_root.resolve()
        self.root = base / session
        if not self.root.resolve().is_relative_to(base):
            raise ValueError("Session escapes data root")
        self.path = self.root / "session.json"

    @contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / ".lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise ValueError("Session is locked; retry after the other operation finishes") from error
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(str(os.getpid()))
            yield
        finally:
            path.unlink(missing_ok=True)

    def initialize(self) -> dict:
        if self.path.exists():
            raise ValueError("Session already exists")
        state = {
            "schema_version": 1,
            "photos": {},
            "stages": {stage: {"status": "pending"} for stage in STAGES},
            "history": [],
        }
        self.save(state)
        return state

    def load(self) -> dict:
        if not self.path.is_file():
            raise ValueError("Session does not exist; run init first")
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported session schema")
        return state

    def save(self, state: dict) -> None:
        atomic_json(self.path, state)

    def resolve(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Artifact escapes session folder")
        return path

    def copy_artifact(self, source: Path, directory: str) -> dict:
        source = source.resolve(strict=True)
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError("Artifact must be a non-empty file")
        checksum = digest(source)
        relative = str(Path(directory) / f"{checksum}{source.suffix.lower()}")
        destination = self.resolve(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or digest(destination) != checksum:
            handle, name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
            os.close(handle)
            try:
                shutil.copyfile(source, name)
                if digest(Path(name)) != checksum:
                    raise ValueError("Source changed during copy; retry intake")
                os.replace(name, destination)
            finally:
                Path(name).unlink(missing_ok=True)
        return {"path": relative, "sha256": checksum}

    def reference_artifact(self, source: Path) -> dict:
        source = source.resolve(strict=True)
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError("Artifact must be a non-empty file")
        if source.is_relative_to(self.root.resolve()):
            return {"path": str(source.relative_to(self.root.resolve())), "sha256": digest(source)}
        return {"path": str(source), "sha256": digest(source), "storage": "external"}

    def artifact_path(self, artifact: dict) -> Path:
        if artifact.get("storage") == "external":
            path = Path(artifact["path"])
            if not path.is_absolute():
                raise ValueError("External reference requires an absolute path")
            return path.resolve()
        return self.resolve(artifact["path"])

    def verify(self, artifact: dict) -> bool:
        path = self.artifact_path(artifact)
        return path.is_file() and digest(path) == artifact["sha256"]


def invalidate(state: dict, stage: str, reason: str) -> None:
    for name in STAGES[STAGES.index(stage):]:
        previous = state["stages"][name]
        if previous.get("status") != "pending":
            state["history"].append({
                "at": datetime.now(timezone.utc).isoformat(),
                "stage": name, "reason": reason, "previous": previous,
            })
        state["stages"][name] = {"status": "pending"}

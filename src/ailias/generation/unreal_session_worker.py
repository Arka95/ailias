"""Editor-local, single-job dispatcher. Compatible with Unreal's embedded Python 3.11."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time


def write_json(path, value):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False)
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.01)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Dispatcher:
    def __init__(self, root, metadata, unreal):
        self.root, self.metadata, self.unreal = root, metadata, unreal
        self.active = None
        self.module = None
        self.seen = None
        self.callback = None
        self.last_poll = 0
        self.console_variables = {}

    def set_console_variables(self, commands):
        if not commands:
            return
        world = self.unreal.get_editor_subsystem(self.unreal.UnrealEditorSubsystem).get_editor_world()
        for command in commands.split(","):
            name, _ = command.split(maxsplit=1)
            self.console_variables[name] = self.unreal.SystemLibrary.get_console_variable_float_value(name)
            self.unreal.SystemLibrary.execute_console_command(world, command)

    def restore_console_variables(self):
        if self.console_variables:
            world = self.unreal.get_editor_subsystem(self.unreal.UnrealEditorSubsystem).get_editor_world()
            for name, value in self.console_variables.items():
                self.unreal.SystemLibrary.execute_console_command(world, f"{name} {value}")
            self.console_variables.clear()

    def publish(self, status, **values):
        write_json(self.root / "state.json", {
            "session_id": self.metadata["session_id"], "status": status, "pid": os.getpid(),
            "job_id": self.active["job_id"] if self.active else self.seen, **values})

    def inspect_completion(self):
        receipt = Path(self.active["request"]["receipt"])
        try:
            result = json.loads(receipt.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if result.get("status") not in {self.active["terminal_status"], "failed"}:
            return
        self.restore_console_variables()
        self.publish("failed" if result["status"] == "failed" else "completed")
        self.active = self.module = None

    def tick(self, delta_seconds):
        if time.monotonic() - self.last_poll < .1:
            return
        self.last_poll = time.monotonic()
        if self.active is not None:
            self.inspect_completion()
            return
        path = self.root / "job.json"
        if not path.exists():
            return
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if (envelope.get("session_id") != self.metadata["session_id"]
                or envelope.get("job_id") == self.seen):
            return
        self.seen = envelope["job_id"]
        if os.path.normcase(envelope["project"]) != os.path.normcase(self.metadata["project"]):
            self.publish("failed", error_type="ProjectMismatch")
            return
        if envelope["operation"] == "session-stop":
            self.publish("stopping")
            self.unreal.unregister_slate_post_tick_callback(self.callback)
            self.unreal.EditorPythonScripting.set_keep_python_script_alive(False)
            return
        self.active = envelope
        self.publish("busy")
        try:
            request = envelope["request"]
            if Path(request["receipt"]).exists():
                raise ValueError("Native receipt already exists")
            self.set_console_variables(envelope.get("console_commands", ""))
            if envelope["worker"] is None:
                if envelope["operation"] != "session-probe":
                    raise ValueError("Unsupported built-in Unreal session operation")
                write_json(Path(request["receipt"]), {
                    "status": "session_probe_completed", "pid": os.getpid(),
                    "session_id": self.metadata["session_id"], "project": self.metadata["project"],
                    "engine": self.unreal.SystemLibrary.get_engine_version()})
            else:
                source = Path(envelope["worker"]).read_bytes()
                if hashlib.sha256(source).hexdigest() != envelope["worker_sha256"]:
                    raise ValueError("Unreal worker changed before dispatch")
                spec = importlib.util.spec_from_file_location("ailias_job_" + envelope["job_id"],
                                                             envelope["worker"])
                self.module = importlib.util.module_from_spec(spec)
                exec(compile(source, envelope["worker"], "exec"), self.module.__dict__)
                self.module.main(request, keep_alive=True)
            self.inspect_completion()
        except Exception as error:
            receipt = Path(envelope["request"]["receipt"])
            try:
                result = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
            except (OSError, ValueError):
                result = {}
            result.update(status="failed", error_type=type(error).__name__,
                          project_may_contain_partial_outputs=True)
            write_json(receipt, result)
            try:
                self.restore_console_variables()
            except Exception:
                result["session_cleanup_failed"] = True
                write_json(receipt, result)
            self.publish("failed", error_type=type(error).__name__)
            self.active = self.module = None


def main():
    import unreal

    metadata_path = Path(os.environ["AILIAS_UNREAL_SESSION"])
    deadline = time.monotonic() + 10
    while not metadata_path.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    project = Path(unreal.Paths.get_project_file_path()).resolve()
    if (os.path.normcase(str(project)) != os.path.normcase(metadata["project"])
            or metadata["process"]["pid"] != os.getpid()):
        raise ValueError("Persistent Unreal bootstrap project/process mismatch")
    if not unreal.SystemLibrary.get_engine_version().startswith("5.8."):
        raise ValueError("Persistent Unreal execution requires the verified UE 5.8 API")
    global DISPATCHER
    DISPATCHER = Dispatcher(metadata_path.parent, metadata, unreal)
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    DISPATCHER.callback = unreal.register_slate_post_tick_callback(DISPATCHER.tick)
    DISPATCHER.publish("idle")


if __name__ == "__main__":
    main()

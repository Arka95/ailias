"""Own one project-local editor; serialize jobs without opening a network listener."""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import math
import msvcrt
import os
from pathlib import Path
import subprocess
import time
import uuid

from ailias.settings import PROJECT_ROOT
from ailias.storage import atomic_json, digest
from ailias.generation.unreal_validation import validate_shader_output

CONFIG = PROJECT_ROOT / "config" / "unreal-session.json"
WORKER = Path(__file__).with_name("unreal_session_worker.py")


def load_settings():
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    if set(settings) != {"mode", "startup_timeout_seconds", "poll_seconds"}:
        raise ValueError("Invalid Unreal session configuration keys")
    if settings["mode"] not in ("persistent", "oneshot"):
        raise ValueError("Unreal execution mode must be persistent or oneshot")
    for key in ("startup_timeout_seconds", "poll_seconds"):
        value = settings[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Unreal session durations must be finite and positive")
    return settings


def process_identity(pid):
    """Include process creation time so a recycled PID cannot acquire ownership."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        *([ctypes.POINTER(wintypes.FILETIME)] * 4),
    ]
    kernel.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() in (87, 1168):
            return None
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        waiting = kernel.WaitForSingleObject(handle, 0)
        if waiting == 0:
            return None
        if waiting != 258:
            raise ctypes.WinError(ctypes.get_last_error())
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(
            handle, *(ctypes.byref(value) for value in times)
        ):
            error = ctypes.get_last_error()
            if kernel.WaitForSingleObject(handle, 100) == 0:
                return None
            raise ctypes.WinError(error)
        name, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
        if not kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
            error = ctypes.get_last_error()
            if kernel.WaitForSingleObject(handle, 100) == 0:
                return None
            raise ctypes.WinError(error)
        return {
            "pid": pid,
            "created": times[0].dwHighDateTime << 32 | times[0].dwLowDateTime,
            "executable": os.path.normcase(str(Path(name.value).resolve())),
        }
    finally:
        kernel.CloseHandle(handle)


class UnrealSession:
    def __init__(self, project, editor=None, settings=None):
        self.project = Path(project).resolve(strict=True)
        if self.project.suffix.lower() != ".uproject":
            raise ValueError("Unreal session requires an existing .uproject")
        self.editor = Path(editor).resolve(strict=True) if editor is not None else None
        self.settings = settings if settings is not None else load_settings()
        self.root = self.project.parent / "Saved" / "AiliasSession"
        self.metadata = self.root / "session.json"
        self.state = self.root / "state.json"
        self.inbox = self.root / "job.json"

    @contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "writer.lock").open("a+b") as stream:
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError(
                    "Unreal project is busy with another ailias operation"
                ) from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

    def status(self):
        if not self.metadata.exists():
            return {"status": "stopped", "project": str(self.project)}
        metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
        if os.path.normcase(
            str(Path(metadata["project"]).resolve())
        ) != os.path.normcase(str(self.project)):
            raise ValueError("Unreal session belongs to another project")
        identity = process_identity(metadata["process"]["pid"])
        if identity != metadata["process"]:
            return {"status": "stopped", "project": str(self.project)}
        if self.editor is not None and metadata["process"][
            "executable"
        ] != os.path.normcase(str(self.editor)):
            raise ValueError("Unreal session belongs to another editor executable")
        state = (
            json.loads(self.state.read_text(encoding="utf-8"))
            if self.state.exists()
            else {}
        )
        if state.get("session_id") != metadata["session_id"]:
            state = {"status": "starting"}
        if self.inbox.exists():
            submitted = json.loads(self.inbox.read_text(encoding="utf-8"))
            if submitted.get("session_id") == metadata["session_id"] and submitted.get(
                "job_id"
            ) != state.get("job_id"):
                state = {"status": "pending", "job_id": submitted["job_id"]}
        return {
            **state,
            "session_id": metadata["session_id"],
            "project": str(self.project),
            "pid": identity["pid"],
            "worker_sha256": metadata["worker_sha256"],
        }

    def assert_idle(self):
        status = self.status()
        if status["status"] not in {"idle", "completed", "stopped"}:
            raise ValueError(
                "Unreal session is busy or unhealthy; inspect session status before retrying"
            )
        return status

    def _start(self):
        current = self.status()
        if current["status"] != "stopped":
            if current.get("worker_sha256") != digest(WORKER):
                raise ValueError(
                    "Unreal session bootstrap changed; stop and restart the owned session"
                )
            return self.assert_idle(), True
        if self.editor is None:
            raise ValueError("Starting an Unreal session requires --editor")
        session_id = uuid.uuid4().hex
        start = time.monotonic()
        environment = {**os.environ, "AILIAS_UNREAL_SESSION": str(self.metadata)}
        log = self.root / "editor.log"
        with log.open("wb") as stream:
            process = subprocess.Popen(
                [
                    str(self.editor),
                    str(self.project),
                    f"-ExecutePythonScript={WORKER}",
                    "-unattended",
                    "-nop4",
                    "-nosplash",
                    "-AllowCommandletRendering",
                    "-RenderOffscreen",
                    "-d3d12",
                    "-sm6",
                    "-stdout",
                    "-FullStdOutLogOutput",
                    "-ExecCmds=mh.Character.FromCustomMeshImportShowIterations 0",
                ],
                cwd=self.project.parent,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        identity = process_identity(process.pid)
        if identity is None:
            raise ValueError(f"Unreal exited during startup; inspect {log}")
        atomic_json(
            self.metadata,
            {
                "session_id": session_id,
                "project": str(self.project),
                "process": identity,
                "worker_sha256": digest(WORKER),
            },
        )
        while time.monotonic() - start < self.settings["startup_timeout_seconds"]:
            current = self.status()
            if current["status"] == "idle":
                return {
                    **current,
                    "startup_seconds": round(time.monotonic() - start, 3),
                }, False
            if current["status"] in {"stopped", "failed"}:
                break
            time.sleep(self.settings["poll_seconds"])
        raise ValueError(
            f"Unreal did not become ready; inspect {log}; stop the owned session before retrying"
        )

    def start(self):
        with self.lock():
            status, reused = self._start()
            return {**status, "reused": reused}

    def run(
        self,
        worker,
        request,
        operation,
        directory,
        terminal_status,
        timeout,
        console_commands="",
    ):
        start = time.monotonic()
        payload = json.loads(Path(request).read_text(encoding="utf-8"))
        receipt = Path(payload["receipt"])
        if not receipt.is_absolute() or not receipt.parent.is_dir():
            raise ValueError(
                "Native receipt requires an absolute path in an existing directory"
            )
        with self.lock():
            status, reused = self._start()
            self.assert_idle()
            engine_log = self.root / "editor.log"
            with engine_log.open("rb") as stream:
                startup = stream.read(1_048_576).decode("utf-8", errors="replace")
            validate_shader_output(startup, require_platform=True)
            log_offset, log_overlap, shader_error = engine_log.stat().st_size, "", None
            if receipt.exists():
                raise ValueError("Refusing to reuse an existing Unreal job receipt")
            worker = Path(worker).resolve(strict=True) if worker else None
            job_id = uuid.uuid4().hex
            envelope = {
                "session_id": status["session_id"],
                "job_id": job_id,
                "operation": operation,
                "project": str(self.project),
                "worker": str(worker) if worker else None,
                "worker_sha256": digest(worker) if worker else None,
                "request": payload,
                "terminal_status": terminal_status,
                "console_commands": console_commands,
            }
            atomic_json(self.inbox, envelope)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                with engine_log.open("rb") as stream:
                    stream.seek(log_offset)
                    while output := stream.read(1_048_576):
                        log_offset += len(output)
                        text = log_overlap + output.decode("utf-8", errors="replace")
                        try:
                            validate_shader_output(text)
                        except ValueError as error:
                            shader_error = error
                        log_overlap = text[-512:]
                current = self.status()
                if current["status"] == "stopped":
                    raise ValueError("Unreal editor exited before the job completed")
                if current.get("job_id") == job_id and current["status"] in {
                    "completed",
                    "failed",
                }:
                    if current["status"] == "failed":
                        raise ValueError(f"Unreal job failed; inspect {receipt}")
                    if not receipt.is_file():
                        raise ValueError("Unreal completed without the native receipt")
                    if shader_error is not None:
                        raise ValueError(
                            f"{shader_error}; inspect {engine_log}"
                        ) from shader_error
                    logs = Path(directory) / "logs"
                    logs.mkdir(parents=True, exist_ok=True)
                    log = logs / f"{operation}.log"
                    result = {
                        "operation": operation,
                        "tool": "unreal",
                        "mode": "persistent",
                        "seconds": round(time.monotonic() - start, 3),
                        "log": str(log),
                        "pid": status["pid"],
                        "session_id": status["session_id"],
                        "job_id": job_id,
                        "reused": reused,
                        "startup_seconds": status.get("startup_seconds", 0),
                    }
                    atomic_json(log, result)
                    return result
                time.sleep(self.settings["poll_seconds"])
            # The editor may still be mutating assets. Do not retry or admit another job.
            raise TimeoutError(
                "Unreal job timed out; session remains busy until native completion"
            )

    def probe(self, directory, timeout=30):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        request = directory / f"probe-{uuid.uuid4().hex}.json"
        receipt = request.with_name(request.stem + "-receipt.json")
        atomic_json(request, {"receipt": str(receipt)})
        runtime = self.run(
            None,
            request,
            "session-probe",
            directory,
            "session_probe_completed",
            timeout,
        )
        return {
            "runtime": runtime,
            "native": json.loads(receipt.read_text(encoding="utf-8")),
        }

    def stop(self, timeout=30):
        with self.lock():
            current = self.status()
            if current["status"] not in {
                "idle",
                "completed",
                "failed",
                "starting",
                "stopped",
            }:
                raise ValueError(
                    "Unreal is still running a job; wait for its completion before stopping"
                )
            result = self._wait_for_exit(current, timeout, shutdown=True)
        return self._finalize_retention(result)

    def wait(self):
        return self._finalize_retention(self._wait_for_exit(self.status(), None))

    def _finalize_retention(self, result):
        from ailias.generation.unreal_retention import finalize_deferred_native

        retention = finalize_deferred_native(self)
        return {**result, "retention": retention} if retention is not None else result

    def _wait_for_exit(self, current, timeout, *, shutdown=False):
        if current["status"] == "stopped":
            return current
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, current["pid"])
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if shutdown:
                atomic_json(
                    self.inbox,
                    {
                        "session_id": current["session_id"],
                        "job_id": uuid.uuid4().hex,
                        "project": str(self.project),
                        "operation": "session-stop",
                    },
                )
            # Retain the handle before shutdown; reopening a terminating PID can return access denied.
            outcome = kernel.WaitForSingleObject(
                handle, int(timeout * 1000) if timeout is not None else 0xFFFFFFFF
            )
            if outcome == 0:
                return {"status": "stopped", "project": str(self.project)}
            if outcome != 258:
                raise ctypes.WinError(ctypes.get_last_error())
            raise TimeoutError("Owned Unreal editor has not completed shutdown")
        finally:
            kernel.CloseHandle(handle)


def execute_unreal(
    tools,
    project,
    worker,
    request,
    operation,
    directory,
    request_variable,
    terminal_status,
    *,
    independent=False,
    console_commands="",
):
    """Use a shared editor when supported; independent reload always uses a new process."""
    runner = getattr(type(tools), "run_unreal", None)
    if runner is not None:
        return runner(
            tools,
            project,
            worker,
            request,
            operation,
            directory,
            request_variable,
            terminal_status,
            independent=independent,
            console_commands=console_commands,
        )
    return tools.run(
        "unreal",
        operation,
        [
            str(project),
            f"-ExecutePythonScript={worker}",
            "-unattended",
            "-nop4",
            "-nosplash",
            "-AllowCommandletRendering",
            "-RenderOffscreen",
            "-d3d12",
            "-sm6",
            "-stdout",
            "-FullStdOutLogOutput",
            *([f"-ExecCmds={console_commands}"] if console_commands else []),
        ],
        directory,
        environment={request_variable: str(request)},
    )

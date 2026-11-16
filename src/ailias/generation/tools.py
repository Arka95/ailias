"""Probe and run shared native tools with explicit timeouts and bounded logs."""

import json
import os
import re
import subprocess
import time
from pathlib import Path

from ailias.generation.settings import TOOL_SPECS

MAX_LOG_BYTES = 1_048_576


class ToolError(ValueError):
    pass


class NativeTools:
    def __init__(self, root: Path, timeout: int = 1200):
        self.specs = json.loads(TOOL_SPECS.read_text(encoding="utf-8"))
        self.paths = {name: root / spec["executable"] for name, spec in self.specs.items()}
        self.paths["colmap"] = self.paths["colmap"].parent / "bin" / "colmap.exe"
        self.timeout = timeout

    def environment(self, name):
        env = os.environ.copy()
        if name == "colmap":
            root = self.paths[name].parent.parent
            env["PATH"] = str(root / "bin") + os.pathsep + env.get("PATH", "")
            env["QT_PLUGIN_PATH"] = str(root / "plugins")
        return env

    def probe(self, name) -> dict:
        path = self.paths[name]
        result = {"path": str(path), "ready": False}
        if not path.is_file():
            return {**result, "reason": "not_installed; run scripts\\setup-generation-tools.ps1"}
        try:
            process = subprocess.run(
                [str(path), "--version" if name == "blender" else "-h"],
                env=self.environment(name), capture_output=True, timeout=30, check=False,
                encoding="utf-8", errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return {**result, "reason": type(error).__name__}
        output = process.stdout + process.stderr
        pattern = r"Blender (4\.5\.\d+)" if name == "blender" else r"COLMAP (4\.2\.\d+)"
        match = re.search(pattern, output)
        ready = process.returncode == 0 and match is not None
        if name == "colmap":
            ready = ready and "with CUDA" in output
        return {**result, "ready": ready, "version": match[1] if match else None,
                "build": output.splitlines()[0] if output.splitlines() else "",
                "reason": None if ready else "unsupported_version_or_missing_CUDA_build"}

    def run(self, name: str, operation: str, args: list, directory: Path,
            environment: dict | None = None) -> dict:
        logs = directory / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{operation}.log"
        start = time.monotonic()
        failure = None
        try:
            with log.open("wb") as stream:
                result = subprocess.run(
                    [str(self.paths[name]), *map(str, args)], cwd=directory,
                    env={**self.environment(name), **(environment or {})},
                    stdout=stream, stderr=subprocess.STDOUT,
                    timeout=self.timeout, check=False,
                )
            if result.returncode:
                failure = f"exit_{result.returncode}"
        except (OSError, subprocess.TimeoutExpired) as error:
            failure = type(error).__name__
        finally:
            if log.exists() and log.stat().st_size > MAX_LOG_BYTES:
                with log.open("rb") as stream:
                    stream.seek(-MAX_LOG_BYTES, os.SEEK_END)
                    tail = stream.read()
                log.write_bytes(b"[earlier native output truncated]\n" + tail)
        if failure:
            raise ToolError(f"{operation}: {failure}; inspect {log}")
        return {"operation": operation, "tool": name,
                "seconds": round(time.monotonic() - start, 3),
                "log": str(log)}

"""Generate selected-profile narration through an existing, unmodified Yapit MCP server."""

import asyncio
import json
import math
import os
import uuid
from pathlib import Path

from ailias.reanimation.media import inspect_audio
from ailias.reanimation.settings import DEFAULT_CONFIG
from ailias.storage import atomic_json

DEFAULT_YAPIT_CONFIG = DEFAULT_CONFIG.with_name("yapit.local.json")


def load_server(path):
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("Yapit configuration must be a JSON object")
    command = settings.get("command")
    arguments = settings.get("args", [])
    environment = settings.get("env", {})
    timeout = settings.get("timeout_seconds", 300)
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Yapit configuration requires the existing MCP server command")
    if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
        raise ValueError("Yapit server args must be a list of strings")
    if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
        raise ValueError("Yapit server env must map strings to strings")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Yapit timeout_seconds must be finite and positive")
    cwd = settings.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise ValueError("Yapit configuration requires the existing server cwd")
    directory = (path.resolve().parent / cwd).resolve()
    if not directory.is_dir():
        raise ValueError("Yapit server cwd does not exist")
    return {"command": command, "args": arguments, "env": environment,
            "cwd": str(directory), "timeout_seconds": timeout}


async def call_yapit(settings, root, request):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    environment = {**os.environ, **settings["env"], "YAPIT_WORKSPACE": str(root / "workspace"),
                   "YAPIT_SPEECH_CACHE": "0"}
    parameters = StdioServerParameters(command=settings["command"], args=settings["args"],
                                      cwd=settings["cwd"], env=environment)
    timeout = settings["timeout_seconds"]
    with (root / "mcp.stderr.log").open("w", encoding="utf-8") as stderr:
        async with asyncio.timeout(timeout):
            async with stdio_client(parameters, errlog=stderr) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timeout) as session:
                    await session.initialize()
                    result = await session.call_tool("generate_audio", request, read_timeout_seconds=timeout)
                    atomic_json(root / "mcp-response.json", result.model_dump(mode="json"))
                    if getattr(result, "is_error", getattr(result, "isError", False)):
                        raise ValueError("Yapit generate_audio returned an error")
                    output = (getattr(result, "structured_content", None)
                              or getattr(result, "structuredContent", None))
                    if output is None:
                        output = json.loads(next(part.text for part in result.content if part.type == "text"))
                    if not isinstance(output, dict):
                        raise ValueError("Yapit generate_audio did not return an artifact object")
                    return output


def generate_narration(store, voice, script_path, config_path=DEFAULT_YAPIT_CONFIG, *,
                       seconds=None, style="default", max_audio_seconds=120):
    if not isinstance(voice, str) or not voice.strip():
        raise ValueError("--voice must name an existing Yapit voice profile")
    if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                                or not math.isfinite(seconds) or not 5 <= seconds <= 180):
        raise ValueError("--seconds must be a finite Yapit target between 5 and 180")
    settings = load_server(config_path)
    script = json.loads(script_path.read_text(encoding="utf-8-sig"))
    if not isinstance(script, dict):
        raise ValueError("--script must contain reviewed Yapit Script JSON, not a writing prompt")
    request = {"script": script, "voice": voice, "style": style}
    if seconds is not None:
        request["seconds"] = seconds
    root = store.resolve(str(Path("reanimation") / "narration" / uuid.uuid4().hex))
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    report = {"status": "running", "voice": voice, "style": style, "transport": "MCP stdio",
              "tool": "generate_audio", "artifacts": {}}
    atomic_json(root / "request.json", request)
    atomic_json(manifest, report)
    try:
        output = asyncio.run(call_yapit(settings, root, request))
        atomic_json(root / "response.json", output)
        directory = str(root.relative_to(store.root) / "artifacts")
        provenance_ref = store.copy_artifact(Path(output["provenance"]), directory)
        provenance = json.loads(store.artifact_path(provenance_ref).read_text(encoding="utf-8-sig"))
        if provenance.get("voice") != voice:
            raise ValueError("Yapit provenance does not match the requested voice")
        audio_ref = store.copy_artifact(Path(output["wav"]), directory)
        audio = store.artifact_path(audio_ref)
        report["audio"] = inspect_audio(audio, max_audio_seconds)
        report.update(status="complete", provenance=provenance)
        report["artifacts"] = {"generated_audio": audio_ref, "voice_provenance": provenance_ref,
                               "narration_request": store.reference_artifact(root / "request.json"),
                               "narration_response": store.reference_artifact(root / "response.json")}
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        if isinstance(error, ImportError):
            raise ImportError("Voice synthesis requires ailias[reanimation-voice] in the host environment") from None
        raise ValueError(f"Yapit narration failed ({type(error).__name__}); inspect {manifest}") from None
    finally:
        atomic_json(manifest, report)
    report["artifacts"]["narration_manifest"] = store.reference_artifact(manifest)
    return {"audio": audio, "voice": voice, "style": style, "artifacts": report["artifacts"]}

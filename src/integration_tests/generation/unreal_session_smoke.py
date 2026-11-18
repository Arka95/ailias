"""Opt-in same-editor probes and optional two-frame MRQ check; never fit or assemble a head."""

import argparse
import json
from pathlib import Path
import uuid

from ailias.generation.unreal_session import UnrealSession, process_identity
from ailias.reanimation.media import inspect_frames
from ailias.reanimation.settings import ReelPolicy
from ailias.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--editor", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--blueprint", help="Existing assembled Blueprint; enables two actual one-frame renders")
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite an existing native session report")
    root = args.report.resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    session = UnrealSession(args.project, args.editor)
    initial = session.start()
    report = {"initial": initial, "probes": [], "renders": [], "status": "running",
              "head_fit_or_assembly_requested": False, "skin_or_likeness_approved": False}
    state = json.loads(session.state.read_text(encoding="utf-8"))
    if state.get("job_id") is None:
        identity = process_identity(initial["pid"])
        created_epoch = identity["created"] / 10_000_000 - 11_644_473_600
        report["process_creation_to_bootstrap_seconds"] = round(
            session.state.stat().st_mtime - created_epoch, 3)
    atomic_json(args.report, report)
    try:
        for _ in range(2):
            report["probes"].append(session.probe(root))
            atomic_json(args.report, report)
        if args.blueprint:
            policy = ReelPolicy(width=180, height=320, temporal_samples=1, warmup_frames=4)
            worker = Path(__file__).parents[2] / "ailias" / "reanimation" / "unreal_worker.py"
            for index in range(2):
                directory = root / f"render-{index}"
                frames = directory / "frames"
                frames.mkdir(parents=True)
                request, receipt = directory / "request.json", directory / "receipt.json"
                atomic_json(request, {
                    "destination": f"/Game/Ailias/SessionChecks/R{uuid.uuid4().hex}",
                    "receipt": str(receipt), "blueprint": args.blueprint, "neutral_review": True,
                    "policy": policy.to_dict(), "framing": policy.framing(), "placement": policy.placement(),
                    "frames": str(frames), "frame_count": 1,
                })
                runtime = session.run(worker, request, f"session-render-{index}", directory,
                                      "frames_rendered", 300,
                                      console_commands="r.DefaultFeature.AutoExposure 0")
                native = json.loads(receipt.read_text(encoding="utf-8"))
                images = inspect_frames(frames, policy, 1)
                if not native.get("render_success") or not native.get("neutral_review"):
                    raise ValueError("One-frame native render was not confirmed")
                report["renders"].append({"runtime": runtime, "native": native,
                                          "frame": str(images[0])})
                atomic_json(args.report, report)
        pids = [value["runtime"]["pid"] for value in report["probes"] + report["renders"]]
        if set(pids) != {initial["pid"]}:
            raise ValueError("Native jobs did not all use the same owned editor")
        if session.status()["status"] != "completed":
            raise ValueError("Reusable editor was not ready after the last job")
        report.update(status="passed", same_pid_verified=True)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(args.report, report)
    print(json.dumps({"status": report["status"], "pid": initial["pid"], "report": str(args.report),
                      "probe_seconds": [item["runtime"]["seconds"] for item in report["probes"]],
                      "render_seconds": [item["runtime"]["seconds"] for item in report["renders"]]}))


if __name__ == "__main__":
    main()

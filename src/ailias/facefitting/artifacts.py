"""Read facefitting revision manifests and verify their artifact dependencies."""

import json
from pathlib import Path

from ailias.imageprocessing.pipeline import component


def inspect_head_fit(store, job):
    path = store.resolve(str(Path("facefitting") / "runs" / component(job) / "manifest.json"))
    report = json.loads(path.read_text(encoding="utf-8"))
    artifacts = [report["inputs"], *report["artifacts"].values(), *report.get("input_dependencies", [])]
    artifacts += [asset["artifact"] for asset in report["assets"].values()]
    problems = [artifact["path"] for artifact in artifacts if not store.verify(artifact)]
    return {**report, "integrity_problems": problems}


def inspect_head_appearance(store, revision):
    path = store.resolve(str(Path("facefitting") / "appearance" / component(revision) / "manifest.json"))
    report = json.loads(path.read_text(encoding="utf-8"))
    dependencies = [report["approval"], report["approved_mesh"], report["fit_manifest"],
                    *report.get("input_dependencies", []), *report["artifacts"].values()]
    return {**report, "integrity_problems": [item["path"] for item in dependencies if not store.verify(item)]}

"""Native package retention through an already-running, owned Unreal editor."""

import json
from pathlib import Path
import subprocess
import uuid

from ailias.generation.unreal_session import UnrealSession
from ailias.resources import is_link
from ailias.resource_management.job_resources import owned_files
from ailias.storage import atomic_json
from ailias.common.file_checksums import FileChecksums


class NativeRetention:
    def __init__(self, store, session):
        self.store, self.session = store, session
        self.directory = store.resolve(
            str(Path("maintenance") / "native-retention" / uuid.uuid4().hex)
        )
        self.pending = session.root / "pending-retention.json"
        self.checksums = FileChecksums()

    def assert_offline(self):
        if self.session.status()["status"] != "stopped":
            raise ValueError("Native file cleanup requires a stopped editor")
        query = (
            "Get-CimInstance Win32_Process -Filter "
            "\"Name = 'UnrealEditor.exe' OR Name = 'UnrealEditor-Cmd.exe'\" "
            "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        process = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", query],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        editors = json.loads(process.stdout) if process.stdout.strip() else []
        if isinstance(editors, dict):
            editors = [editors]
        for editor in editors:
            command = editor.get("CommandLine")
            if (
                not command
                or self.session.project.name.casefold() in command.casefold()
            ):
                raise ValueError(
                    "Another editor may have this project open; native cleanup is deferred"
                )

    def package_files(self):
        return {
            path
            for path in owned_files(self.session.project.parent / "Content")
            if path.suffix.lower() in {".uasset", ".umap"}
        }

    def validate_snapshot(self, plan):
        self.assert_offline()
        if Path(plan["project"]) != self.session.project:
            raise ValueError("Retention project changed")
        snapshot = {
            Path(path): checksum for path, checksum in plan["package_snapshot"].items()
        }
        current = self.package_files()
        absent = set(snapshot) - current
        if current - set(snapshot) or absent - {
            Path(item["path"]) for item in plan["delete"]
        }:
            raise ValueError(
                "Native package inventory changed; inspect it in Unreal before pruning"
            )
        if any(self.checksums(path) != snapshot[path] for path in current):
            raise ValueError(
                "Native package content changed; inspect it in Unreal before pruning"
            )

    def inspection(self):
        if self.session.status()["status"] != "stopped":
            return self.run({"operation": "inspect"})
        saved = json.loads(self.pending.read_text(encoding="utf-8"))
        if (
            saved["alias"] != self.store.root.name
            or Path(saved["data_root"]) != self.store.root.parent
        ):
            raise ValueError("Native retention snapshot belongs to another alias")
        plan = saved["plan"]
        self.validate_snapshot(plan)
        content = self.session.project.parent / "Content"
        previous = plan["graph"]
        graph = {
            package: links
            for package, links in previous.items()
            if any(
                (
                    content
                    / (package.removeprefix("/Game/") + suffix).replace("/", "\\")
                ).is_file()
                for suffix in (".uasset", ".umap")
            )
        }
        missing = set(previous) - set(graph)
        return {
            "graph": {
                package: {
                    kind: [value for value in links[kind] if value not in missing]
                    for kind in ("dependencies", "referencers")
                }
                for package, links in graph.items()
            },
            "dirty_packages": plan.get("dirty_packages", []),
        }

    def run(self, payload):
        status = self.session.assert_idle()
        if status["status"] == "stopped":
            raise ValueError("Retention never starts an editor")
        self.directory.mkdir(parents=True, exist_ok=True)
        request = self.directory / f"{uuid.uuid4().hex}.json"
        receipt = request.with_name(request.stem + "-receipt.json")
        atomic_json(request, {**payload, "receipt": str(receipt)})
        self.session.run(
            Path(__file__).with_name("unreal_retention_worker.py"),
            request,
            "native-retention",
            self.directory,
            "retention_completed",
            300,
        )
        return json.loads(receipt.read_text(encoding="utf-8"))

    def plan(self, keep, documents, known):
        self.checksums = digest = FileChecksums()
        inspection = self.inspection()
        graph = inspection["graph"]
        content = self.session.project.parent / "Content"
        if any(
            path.is_relative_to(content)
            and path.suffix in {".uasset", ".umap"}
            and not path.is_file()
            for path in keep
        ):
            raise ValueError(
                "A live native output is missing; retain existing packages for recovery"
            )
        files = {
            package: content
            / (package.removeprefix("/Game/") + ".uasset").replace("/", "\\")
            for package in graph
        }
        owned = {
            package
            for package, path in files.items()
            if path in known
            and path.is_file()
            and not is_link(path)
            and digest(path) in known[path]
        }
        roots = {
            package
            for package, path in files.items()
            if path in keep or package not in owned
        }
        roots.update(inspection["dirty_packages"])
        for package, links in graph.items():
            if any(
                not value.startswith("/Game/Ailias/") for value in links["referencers"]
            ):
                roots.add(package)
        from ailias.common.json_values import iter_nested_strings

        for value in documents.values():
            for text in iter_nested_strings(value):
                package = text.split(".", 1)[0]
                if package in graph:
                    roots.add(package)
        pending, live = list(roots), set()
        while pending:
            package = pending.pop()
            if package in live:
                continue
            live.add(package)
            pending.extend(
                dep
                for dep in graph.get(package, {}).get("dependencies", [])
                if dep in graph
            )
        delete = []
        for package in sorted(owned - live):
            path = files[package]
            if any(
                is_link(parent)
                for parent in path.parents
                if parent.is_relative_to(content)
            ):
                raise ValueError("Native retention cannot traverse links")
            stat = path.stat()
            delete.append(
                {
                    "package": package,
                    "path": str(path),
                    "sha256": digest(path),
                    "bytes": stat.st_size if stat.st_nlink == 1 else 0,
                }
            )
        protected = [
            {"path": str(files[package]), "sha256": digest(files[package])}
            for package in live
            if package in files and files[package].is_file()
        ]
        return {
            "project": str(self.session.project),
            "graph": graph,
            "dirty_packages": inspection["dirty_packages"],
            "delete": delete,
            "protected": protected,
            "kept_files": [str(files[package]) for package in live if package in files],
            "planned_bytes": sum(item["bytes"] for item in delete),
            "package_snapshot": {
                str(path): digest(path) for path in self.package_files()
            },
        }

    def apply(self, plan):
        if str(self.session.project) != plan["project"]:
            raise ValueError("Retention project changed")
        self.session.assert_idle()
        result = {"removed_files": 0, "removed_bytes": 0, "errors": 0}
        if self.session.status()["status"] != "stopped":
            atomic_json(
                self.pending,
                {
                    "data_root": str(self.store.root.parent),
                    "alias": self.store.root.name,
                    "plan": plan,
                },
            )
            return {
                **result,
                "deferred_files": len(plan["delete"]),
                "deferred_until": "editor_stopped",
            }
        self.validate_snapshot(plan)
        content = self.session.project.parent / "Content" / "Ailias"
        with self.session.lock():
            self.assert_offline()
            for item in plan["delete"]:
                path = Path(item["path"])
                package = item["package"]
                expected = content.parent / (
                    package.removeprefix("/Game/") + ".uasset"
                ).replace("/", "\\")
                if (
                    not package.startswith("/Game/Ailias/")
                    or package not in plan["graph"]
                    or package in plan.get("dirty_packages", [])
                    or set(plan["graph"][package]["referencers"])
                    - {entry["package"] for entry in plan["delete"]}
                    or path != expected
                    or plan["package_snapshot"].get(str(path)) != item["sha256"]
                    or not path.is_relative_to(content)
                    or is_link(path)
                    or any(
                        is_link(parent)
                        for parent in path.parents
                        if parent.is_relative_to(content)
                    )
                    or path.with_suffix(".uexp").exists()
                    or path.with_suffix(".ubulk").exists()
                ):
                    raise ValueError(
                        "Native cleanup requires an exact owned unsplit package"
                    )
            for item in plan["delete"]:
                path = Path(item["path"])
                try:
                    path.unlink()
                    result["removed_files"] += 1
                    result["removed_bytes"] += item["bytes"]
                except OSError:
                    result["errors"] += 1
        return result


def available_native_retention(store, report=None):
    from ailias.resource_management.catalog import ArtifactCatalog

    reports = ([report] if report else []) + [
        value for _, value in ArtifactCatalog(store).discover_jobs().values()
    ]
    seen = set()
    for value in reports:
        project = value.get("readiness", {}).get("project")
        if not project or project in seen or not Path(project).is_file():
            continue
        seen.add(project)
        if not Path(project).is_relative_to(store.root):
            continue
        session = UnrealSession(project)
        status = session.status()["status"]
        if status not in {"idle", "completed", "stopped"}:
            raise ValueError(
                "An owned native operation is active or unhealthy; retention is deferred"
            )
        if status in {"idle", "completed"} or (
            status == "stopped" and (session.root / "pending-retention.json").is_file()
        ):
            return NativeRetention(store, session)
    return None


def finalize_deferred_native(session):
    """Run after natural/explicit shutdown; cleanup itself never closes an editor."""
    pending = session.root / "pending-retention.json"
    if not pending.is_file():
        return None
    try:
        from ailias.resource_management.manager import ResourceManager
        from ailias.storage import Store

        owner = json.loads(pending.read_text(encoding="utf-8"))
        store = Store(Path(owner["data_root"]), owner["alias"])
        if not session.project.is_relative_to(store.root):
            raise ValueError("Native project is outside its owning alias")
        return ResourceManager(store, NativeRetention(store, session)).cleanup(
            apply=True
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result = {"deferred": True, "error_type": type(error).__name__}
        atomic_json(session.root / "retention-deferred.json", result)
        return result

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ailias.generation.unreal import asset_path, readiness, run_unreal
from ailias.generation.unreal_worker import run
from ailias.storage import Store, atomic_json


class UnrealReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.engine = self.root / "Engine"
        self.editor = self.engine / "Binaries" / "Win64" / "UnrealEditor-Cmd.exe"
        self.editor.parent.mkdir(parents=True)
        self.editor.write_bytes(b"fixture, never executed")
        self.project = self.root / "Project" / "Ailias.uproject"
        atomic_json(self.engine / "Build" / "Build.version",
                    {"MajorVersion": 5, "MinorVersion": 8, "PatchVersion": 0})
        for plugin in ("PythonScriptPlugin", "MetaHumanCharacter"):
            atomic_json(self.engine / "Plugins" / plugin / f"{plugin}.uplugin", {"EnabledByDefault": False})
        atomic_json(self.project, {"Plugins": [
            {"Name": name, "Enabled": True} for name in ("PythonScriptPlugin", "MetaHumanCharacter")]})

    def test_requires_existing_editor_matching_version_and_enabled_plugins(self):
        self.assertTrue(readiness(self.editor, self.project)["ready"])
        atomic_json(self.project, {"Plugins": []})
        result = readiness(self.editor, self.project)
        self.assertFalse(result["ready"])
        self.assertIn("Install/enable PythonScriptPlugin", result["blockers"])
        atomic_json(self.engine / "Build" / "Build.version",
                    {"MajorVersion": 5, "MinorVersion": 7, "PatchVersion": 4})
        self.assertTrue(any("UE 5.8" in b for b in readiness(self.editor, self.project)["blockers"]))

    def test_existing_project_is_not_silently_upgraded(self):
        atomic_json(self.project, {"EngineAssociation": "5.7", "Plugins": [
            {"Name": name, "Enabled": True} for name in ("PythonScriptPlugin", "MetaHumanCharacter")]})
        result = readiness(self.editor, self.project)
        self.assertFalse(result["ready"])
        self.assertTrue(any("not silently upgrade" in blocker for blocker in result["blockers"]))

    def test_missing_core_data_warns_without_blocking_plain_mesh_import_readiness(self):
        result = readiness(self.editor, self.project)
        self.assertTrue(result["ready"])
        self.assertFalse(result["metahuman_core_data"]["installed"])
        self.assertTrue(any("Core Data" in warning for warning in result["warnings"]))

    def test_core_data_requires_model_and_body_texture_payloads_not_just_folders(self):
        content = self.engine / "Plugins" / "MetaHumanCharacter" / "Content" / "Optional"
        synthesis, body = content / "TextureSynthesis" / "models", content / "BodyTextures"
        synthesis.mkdir(parents=True)
        body.mkdir(parents=True)
        self.assertFalse(readiness(self.editor, self.project)["metahuman_core_data"]["installed"])
        (synthesis / "fixture.ar").write_bytes(b"fixture, never loaded")
        self.assertFalse(readiness(self.editor, self.project)["metahuman_core_data"]["installed"])
        (body / "fixture.uasset").write_bytes(b"fixture, never loaded")
        result = readiness(self.editor, self.project)
        self.assertTrue(result["metahuman_core_data"]["installed"])
        self.assertEqual(result["warnings"], [])

    def test_missing_editor_does_not_download_or_launch_anything(self):
        with patch("ailias.generation.unreal.shutil.which", return_value=None):
            self.assertFalse(readiness(None, None)["ready"])
        with self.assertRaises(ValueError):
            asset_path("../Content/UserAsset")

    def test_no_editor_or_cleaned_mesh_records_blocked_without_placeholder_outputs(self):
        store = Store(self.root / "data", "subject")
        job = "fixture-job"
        inputs = store.resolve("generation\\inputs\\fixture\\sources.json")
        atomic_json(inputs, {"files": [], "inputs": {"entries": []}})
        curation = store.resolve("imageprocessing\\curation.json")
        atomic_json(curation, {"fixture": True})
        path = store.resolve(f"generation\\runs\\{job}\\manifest.json")
        atomic_json(path, {
            "schema_version": 1, "job": job, "status": "blocked", "artifacts": {},
            "steps": [], "input_view": store.reference_artifact(inputs),
            "curation": store.reference_artifact(curation),
        })
        tools = Mock()
        with patch("ailias.generation.unreal.shutil.which", return_value=None):
            result = run_unreal(store, job, "import", None, None, tools)
        self.assertEqual(result["metahuman"]["status"], "blocked")
        tools.run.assert_not_called()
        self.assertFalse((path.parent / "unreal").exists())
        self.assertFalse(store.path.exists())

    def imported_job(self, save_asset):
        store = Store(self.root / "data", "subject")
        job = "fixture-job"
        inputs = store.resolve("generation\\inputs\\fixture\\sources.json")
        atomic_json(inputs, {"files": [], "inputs": {"entries": []}})
        curation = store.resolve("imageprocessing\\curation.json")
        atomic_json(curation, {"fixture": True})
        mesh = store.resolve("generation\\runs\\fixture-job\\blender\\head.obj")
        mesh.parent.mkdir(parents=True)
        mesh.write_text("fixture, never imported")
        path = mesh.parent.parent / "manifest.json"
        atomic_json(path, {
            "schema_version": 1, "job": job, "status": "complete",
            "artifacts": {"cleaned_mesh": store.reference_artifact(mesh)},
            "steps": [], "input_view": store.reference_artifact(inputs),
            "curation": store.reference_artifact(curation),
            "metahuman": {"status": "blocked", "blockers": ["previous readiness check"]},
        })
        asset = self.project.parent / "Content" / "Ailias" / "subject" / "fixture_job" / "Head.uasset"

        def launch(name, operation, args, directory, environment):
            request = json.loads(Path(environment["AILIAS_UNREAL_REQUEST"]).read_text())
            if save_asset:
                asset.parent.mkdir(parents=True)
                asset.write_bytes(b"synthetic package, never loaded by Unreal")
                asset.with_suffix(".ubulk").write_bytes(b"synthetic bulk companion")
            atomic_json(Path(request["receipt"]), {
                "status": "mesh_imported", "assets": [request["destination"] + "/Head"],
                "assembled": False, "rig_verified": False,
            })
            return {"operation": operation, "tool": name, "seconds": 0}

        tools = SimpleNamespace(paths={}, run=Mock(side_effect=launch))
        result = run_unreal(store, job, "import", self.editor, self.project, tools)
        return store, path, asset, result

    def test_import_hashes_external_saved_packages_without_cleaning_the_project(self):
        store, path, asset, result = self.imported_job(save_asset=True)
        self.assertEqual(result["metahuman"]["status"], "mesh_imported")
        self.assertTrue(store.verify(result["artifacts"]["unreal_import_asset_0"]))
        self.assertTrue(store.verify(result["artifacts"]["unreal_import_asset_0.ubulk"]))
        self.assertTrue(asset.is_file())
        self.assertEqual(result["unreal_history"][0]["status"], "blocked")
        self.assertEqual(result["cleanup"]["errors"], 0)
        self.assertFalse((path.parent / "unreal-request.json").exists())
        self.assertEqual(json.loads(path.read_text())["metahuman"]["status"], "mesh_imported")

    def test_receipt_without_saved_package_is_failure_not_metahuman_success(self):
        store, path, asset, result = self.imported_job(save_asset=False)
        self.assertEqual(result["metahuman"]["status"], "failed")
        self.assertTrue(result["metahuman"]["project_may_contain_partial_outputs"])
        self.assertFalse(asset.exists())
        self.assertNotIn("unreal_import_asset_0", result["artifacts"])
        self.assertTrue(store.verify(result["artifacts"]["unreal_import_receipt"]))
        self.assertEqual(result["cleanup"]["errors"], 0)
        self.assertFalse((path.parent / "unreal-request.json").exists())


class UnrealWorkerContractTests(unittest.TestCase):
    """Fake UE contracts, not evidence that an engine/MetaHuman ran locally."""

    def fixture(self):
        class Character:
            pass

        class Identity:
            pass

        character, identity = Character(), Identity()
        subsystem = Mock()
        subsystem.try_add_object_to_edit.return_value = True
        subsystem.import_from_identity.return_value = "success"
        library = Mock()
        library.save_loaded_asset.return_value = True
        unreal = SimpleNamespace(
            SystemLibrary=SimpleNamespace(get_engine_version=lambda: "5.8.0-fixture"),
            MetaHumanCharacter=Character, MetaHumanIdentity=Identity,
            MetaHumanCharacterEditorSubsystem=object,
            get_editor_subsystem=lambda cls: subsystem,
            load_asset=lambda name: character if name == "/Game/Character" else identity,
            ImportFromIdentityParams=SimpleNamespace,
            ImportErrorCode=SimpleNamespace(SUCCESS="success"),
            EditorAssetLibrary=library,
        )
        request = {"operation": "conform", "character": "/Game/Character", "identity": "/Game/Identity",
                   "cloud_services": False}
        return unreal, subsystem, request

    def test_conform_checks_return_code_saves_and_releases_owned_editor_state(self):
        unreal, subsystem, request = self.fixture()
        result = run(request, unreal)
        self.assertEqual(result["status"], "identity_conformed")
        self.assertFalse(result["assembled"])
        parameters = subsystem.import_from_identity.call_args.args[2]
        self.assertFalse(parameters.use_metric_scale)
        unreal.EditorAssetLibrary.save_loaded_asset.assert_called_once()
        subsystem.remove_object_to_edit.assert_called_once()
        subsystem.request_auto_rigging.assert_not_called()
        subsystem.request_texture_sources.assert_not_called()

    def test_unconformed_identity_never_commits_or_requests_cloud_and_releases_state(self):
        unreal, subsystem, request = self.fixture()
        subsystem.import_from_identity.return_value = "identity_not_conformed"
        with self.assertRaisesRegex(RuntimeError, "Identity conform failed"):
            run(request, unreal)
        subsystem.commit_face_state.assert_not_called()
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()
        subsystem.request_auto_rigging.assert_not_called()
        subsystem.remove_object_to_edit.assert_called_once()

    def test_cloud_requests_are_explicit_blocking_and_do_not_claim_assembly_or_rig_validation(self):
        unreal, subsystem, request = self.fixture()
        unreal.MetaHumanCharacterAutoRiggingRequestParams = SimpleNamespace
        unreal.MetaHumanCharacterTextureRequestParams = SimpleNamespace
        unreal.MetaHumanRigType = SimpleNamespace(JOINTS_AND_BLENDSHAPES="face_and_blendshapes")
        request["cloud_services"] = True
        result = run(request, unreal)
        self.assertEqual(result["status"], "identity_conformed_cloud_requests_returned")
        self.assertTrue(subsystem.request_auto_rigging.call_args.args[1].blocking)
        self.assertTrue(subsystem.request_texture_sources.call_args.args[1].blocking)
        self.assertFalse(result["assembled"])
        self.assertFalse(result["rig_verified"])

    def test_static_mesh_import_is_not_reported_as_metahuman_generation(self):
        class NativeArray:
            def __iter__(self):
                return iter(("/Game/Fixture/Head", "/Game/Fixture/Texture"))

        class Mesh:
            pass

        class Texture:
            pass

        library = Mock()
        library.list_assets.side_effect = [[], NativeArray()]
        assets = {"/Game/Fixture/Head": Mesh(), "/Game/Fixture/Texture": Texture()}
        unreal = SimpleNamespace(
            SystemLibrary=SimpleNamespace(get_engine_version=lambda: "5.8.0-fixture"),
            EditorAssetLibrary=library, AssetImportTask=SimpleNamespace,
            FbxImportUI=lambda: SimpleNamespace(static_mesh_import_data=SimpleNamespace()),
            FBXImportType=SimpleNamespace(FBXIT_STATIC_MESH="static_mesh"),
            AssetToolsHelpers=SimpleNamespace(get_asset_tools=lambda: Mock()),
            load_asset=assets.get, StaticMesh=Mesh, Texture2D=Texture,
        )
        result = run({"operation": "import", "destination": "/Game/Fixture", "mesh": "head.obj"}, unreal)
        self.assertEqual(result["status"], "mesh_imported")
        self.assertTrue(result["requires_manual_identity_solve"])
        self.assertFalse(result["assembled"])
        self.assertEqual(json.loads(json.dumps(result))["assets"],
                         ["/Game/Fixture/Head", "/Game/Fixture/Texture"])


if __name__ == "__main__":
    unittest.main()

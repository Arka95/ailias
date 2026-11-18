"""Verify native-topology, appearance persistence and no-cloud recovery contracts with a fake editor."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ailias.facefitting.unreal.wrapped_head_import_worker import run


class LikenessImportTests(unittest.TestCase):
    def fixture(self):
        class Asset:
            def __init__(self, name):
                self.name = name

            def get_name(self):
                return self.name

            def get_path_name(self):
                return "/Game/Recovery/" + self.name

        class Mesh(Asset):
            pass

        class Texture(Asset):
            pass

        class Character(Asset):
            def __init__(self):
                super().__init__("PhotoHead")
                overrides = SimpleNamespace(material_overrides=SimpleNamespace(),
                                            enable_material_overrides=False)
                self.settings = SimpleNamespace(texture_material_overrides=overrides)

            def get_editor_property(self, name):
                return self.settings

            def set_editor_property(self, name, value):
                self.settings = value

        mesh, texture, character = Mesh("WrappedNativeHead"), Texture("PhotoColour"), Character()
        instance = Asset("PhotoMaterialInstance")
        tools = Mock()
        tools.create_asset.side_effect = lambda name, *args: instance if name == instance.name else Asset(name)

        def import_assets(tasks):
            tasks[0].imported_object_paths = ["mesh"]
            tasks[1].imported_object_paths = ["texture"]

        tools.import_asset_tasks.side_effect = import_assets
        assets = Mock()
        assets.list_assets.return_value = []
        assets.duplicate_asset.return_value = character
        assets.save_loaded_asset.return_value = True
        subsystem, face, actor = Mock(), Mock(), Mock()
        subsystem.try_add_object_to_edit.return_value = True
        subsystem.import_from_template.return_value = SimpleNamespace(name="SUCCESS")
        subsystem.spawn_meta_human_actor.return_value = actor
        actor.get_components_by_class.return_value = [face]
        face.get_name.return_value = "Face"
        face.get_material.return_value = instance
        face.get_skeletal_mesh_asset.return_value = mesh
        actor_subsystem = Mock()
        editor_subsystem_class, actor_subsystem_class = object(), object()
        unreal = SimpleNamespace(
            EditorAssetLibrary=assets, AssetToolsHelpers=SimpleNamespace(get_asset_tools=lambda: tools),
            AssetImportTask=SimpleNamespace, AssetExportTask=SimpleNamespace, FbxFactory=SimpleNamespace,
            FbxImportUI=lambda: SimpleNamespace(static_mesh_import_data=SimpleNamespace()),
            FBXImportType=SimpleNamespace(FBXIT_STATIC_MESH="static"),
            load_asset=lambda path: {"mesh": mesh, "texture": texture}[path],
            StaticMesh=Mesh, Texture2D=Texture, MetaHumanCharacter=Character,
            Material=Asset, MaterialFactoryNew=SimpleNamespace, MaterialEditingLibrary=Mock(),
            MaterialExpressionTextureSampleParameter2D=object,
            MaterialExpressionConstant=object,
            MaterialProperty=SimpleNamespace(MP_BASE_COLOR="colour", MP_ROUGHNESS="roughness"),
            MaterialInstanceConstant=Asset, MaterialInstanceConstantFactoryNew=SimpleNamespace,
            MetaHumanCharacterEditorSubsystem=editor_subsystem_class,
            EditorActorSubsystem=actor_subsystem_class,
            get_editor_subsystem=lambda cls: subsystem if cls is editor_subsystem_class else actor_subsystem,
            ImportFromTemplateParams=SimpleNamespace,
            MetaHumanAlignmentOptions=SimpleNamespace(NONE="none"),
            MetaHumanCharacterSkinMaterialSlot=SimpleNamespace(
                **{name: name for name in ("LOD0", "LOD1", "LOD2", "LOD3", "LOD4", "LOD5TO7")}),
            SkeletalMeshComponent=object, FbxExportOption=SimpleNamespace,
            Exporter=SimpleNamespace(run_asset_export_task=Mock(return_value=True)),
        )
        request = {"destination": "/Game/Recovery", "character": "/Game/Rejected",
                   "template": "native-topology.obj", "texture": "rebaked-native-uv.png",
                   "native_fbx": "evaluated-output.fbx"}
        return unreal, subsystem, face, character, request

    def test_native_template_uvs_and_persistent_material_are_required_without_cloud_or_body_overwrite(self):
        unreal, subsystem, _, character, request = self.fixture()
        result = run(request, unreal, lambda value: None)
        params = subsystem.import_from_template.call_args.args[-1]
        self.assertTrue(params.match_vertices_by_u_vs)
        self.assertTrue(params.isolate_head_from_body)
        self.assertEqual(params.alignment_options, "none")
        self.assertTrue(character.settings.texture_material_overrides.enable_material_overrides)
        self.assertTrue(result["photo_material_active"])
        self.assertFalse(result["likeness_approved"])
        subsystem.commit_body_state.assert_not_called()
        subsystem.request_auto_rigging.assert_not_called()
        subsystem.request_texture_sources.assert_not_called()
        self.assertEqual(subsystem.remove_object_to_edit.call_count, 2)

    def test_invalid_native_uv_template_never_commits_or_exports_a_claimed_result(self):
        unreal, subsystem, _, _, request = self.fixture()
        subsystem.import_from_template.return_value = SimpleNamespace(name="INVALID_HEAD_MESH")
        with self.assertRaisesRegex(RuntimeError, "template import failed"):
            run(request, unreal, lambda value: None)
        subsystem.commit_face_state.assert_not_called()
        unreal.Exporter.run_asset_export_task.assert_not_called()
        subsystem.remove_object_to_edit.assert_called_once()

    def test_missing_reloaded_photo_material_is_not_published_as_a_textured_character(self):
        unreal, subsystem, face, _, request = self.fixture()
        face.get_material.return_value = SimpleNamespace(get_path_name=lambda: "/Engine/Gray")
        with self.assertRaisesRegex(RuntimeError, "did not activate"):
            run(request, unreal, lambda value: None)
        unreal.Exporter.run_asset_export_task.assert_not_called()
        self.assertEqual(subsystem.remove_object_to_edit.call_count, 2)


if __name__ == "__main__":
    unittest.main()

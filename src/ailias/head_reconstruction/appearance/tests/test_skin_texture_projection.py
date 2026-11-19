"""Native UV preservation and expression offsets without models, GPU or Unreal."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ailias.head_reconstruction.appearance.skin_texture_projection import (
    SkinProjectionMesh,
    bake_photo_skin_texture,
    estimate_photo_alignment,
    rasterize_unreal_uvs,
)
from ailias.head_reconstruction.geometry.tests.fitting_fixtures import make_head_fit
from ailias.storage import Store
from ailias.metahuman.skin_texture_worker import (
    export_skin_uv_mesh,
    import_base_color_texture,
)


class NativePhotoSurfaceTests(unittest.TestCase):
    def test_native_low_v_texels_are_at_the_top_of_the_output_image_not_the_bottom(
        self,
    ):
        uvs = np.array([[[0.1, 0.1], [0.4, 0.1], [0.1, 0.4]]])
        original = uvs.copy()
        rows, _, _, _ = rasterize_unreal_uvs(uvs, 32)
        self.assertLess(rows.max(), 16)
        np.testing.assert_array_equal(uvs, original)

    def fixture(self):
        vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
        triangles = np.array([[0, 1, 2]])
        model = SimpleNamespace(
            neutral=lambda identity: vertices,
            triangles=triangles,
            center=vertices.mean(0),
            expression=np.broadcast_to([0.0, 0.1, 0], (2, 3, 3)).copy(),
        )
        surface = {
            "vertices": (vertices[:, [0, 2, 1]] * 100).tolist(),
            "triangles": triangles.tolist(),
            "triangle_uvs": [[[0.0, 0], [1.0, 0], [0.0, 1]]],
        }
        return surface, model

    def test_native_rest_vertices_and_uv_correspondence_are_not_deformed(self):
        surface, model = self.fixture()
        projected = SkinProjectionMesh(surface, model, np.zeros(0))
        np.testing.assert_array_equal(projected.neutral(np.zeros(0)), model.neutral([]))
        np.testing.assert_array_equal(projected.uvs, surface["triangle_uvs"])
        np.testing.assert_allclose(projected.expression, model.expression)

    def test_reversed_native_winding_keeps_the_correct_uv_at_every_triangle_corner(
        self,
    ):
        surface, model = self.fixture()
        surface["triangles"] = [[0, 2, 1]]
        surface["triangle_uvs"] = [[[0.0, 0], [0.0, 1], [1.0, 0]]]
        projected = SkinProjectionMesh(surface, model, np.zeros(0))
        self.assertTrue(projected.winding_reversed_for_projection)
        np.testing.assert_array_equal(projected.triangles, [[0, 1, 2]])
        np.testing.assert_array_equal(projected.uvs, [[[0.0, 0], [1.0, 0], [0.0, 1]]])

    def test_invalid_native_uv_topology_is_rejected(self):
        surface, model = self.fixture()
        surface["triangle_uvs"] = []
        with self.assertRaisesRegex(ValueError, "matching native UVs"):
            SkinProjectionMesh(surface, model, np.zeros(0))

    def test_canonical_head_height_and_rotation_are_undone_only_for_photo_projection(
        self,
    ):
        surface, model = self.fixture()
        original = np.asarray(surface["vertices"])
        rotation = np.array([[0.0, -1, 0], [1.0, 0, 0], [0.0, 0, 1]])
        canonical = original @ rotation.T + [0, 0, 128]
        alignment = estimate_photo_alignment(
            {
                "face_landmarks_solved_in_target_pose": original.tolist(),
                "face_landmarks_after": canonical.tolist(),
            }
        )
        surface.update(vertices=canonical.tolist(), canonical_to_target=alignment)
        projected = SkinProjectionMesh(surface, model, np.zeros(0))
        np.testing.assert_allclose(projected.neutral([]), model.neutral([]), atol=1e-10)
        np.testing.assert_array_equal(surface["vertices"], canonical)
        self.assertLess(max(alignment["nonrigid_residual_cm_p50_p95_max"]), 1e-10)

    def test_invalid_pose_alignment_cannot_warp_native_geometry_for_projection(self):
        surface, model = self.fixture()
        surface["canonical_to_target"] = {
            "rotation": (np.eye(3) * 2).tolist(),
            "translation_cm": [0, 0, 0],
        }
        with self.assertRaisesRegex(ValueError, "rigid transform"):
            SkinProjectionMesh(surface, model, np.zeros(0))

    def test_typed_fit_projects_colour_without_changing_the_saved_native_surface(self):
        surface, model = self.fixture()
        fitted = make_head_fit(["front"], identity=np.array([0.25]))
        identities = []
        vertices = model.neutral([])

        def neutral(identity):
            identities.append(identity.copy())
            return vertices

        model.neutral = neutral

        def project(store, root, surface, identity, mapping, *args, **kwargs):
            count = len(mapping[0])
            return {
                "skin_color": np.full(3, 0.4),
                "color": np.full((count, 3), 0.5),
                "coverage": np.ones(count, dtype=bool),
                "hair_color": np.full(3, 0.1),
                "hair_length": 0.02,
                "report": {"source": "synthetic"},
            }, [{"face_id": "front"}]

        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory), "subject")
            store.root.mkdir()
            source = store.root / "surface.json"
            original = json.dumps(surface)
            source.write_text(original, encoding="utf-8")
            with patch(
                "ailias.head_reconstruction.appearance.skin_texture_projection.project_photo_evidence",
                side_effect=project,
            ):
                result = bake_photo_skin_texture(
                    store,
                    store.root,
                    source,
                    model,
                    fitted,
                    [],
                    {},
                    "cpu",
                    texture_size=8,
                )
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            with Image.open(result["texture"]) as texture:
                self.assertEqual(texture.size, (8, 8))
            self.assertEqual(result["selected_views"], ["front"])
            self.assertTrue(
                all(np.array_equal(identity, [0.25]) for identity in identities)
            )


class NativeUvWorkerTests(unittest.TestCase):
    def fixture(self):
        class Mesh:
            def get_editor_property(self, name):
                return [
                    SimpleNamespace(material_slot_name="Head", material_interface=None)
                ]

        vector = lambda x, y, z=0: SimpleNamespace(x=x, y=y, z=z)
        queries = SimpleNamespace(
            get_num_triangle_i_ds=lambda mesh: 2,
            get_triangle_indices=lambda mesh, triangle: (vector(0, 1, 2), True),
            get_triangle_positions=lambda mesh, triangle: (
                True,
                vector(0, 0),
                vector(1, 0),
                vector(0, 1),
            ),
            get_triangle_u_vs=lambda mesh, channel, triangle: (
                vector(0, 0),
                vector(1, 0),
                vector(0.25 * triangle, 1),
                True,
            ),
        )
        unreal = SimpleNamespace(
            load_asset=lambda name: Mesh(),
            SkeletalMesh=Mesh,
            DynamicMesh=object,
            GeometryScript_AssetUtils=SimpleNamespace(
                copy_mesh_from_skeletal_mesh=lambda *args: (object(), "success")
            ),
            GeometryScriptCopyMeshFromAssetOptions=object,
            GeometryScriptMeshReadLOD=object,
            GeometryScriptOutcomePins=SimpleNamespace(SUCCESS="success"),
            GeometryScript_MeshQueries=queries,
            GeometryScript_Materials=SimpleNamespace(
                get_triangle_material_id=lambda *args: (0, True)
            ),
            EditorAssetLibrary=Mock(),
            AssetToolsHelpers=Mock(),
        )
        return unreal

    def test_uv_seams_are_exported_per_corner_without_duplicating_native_vertices(self):
        unreal = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "native.json"
            result = export_skin_uv_mesh(
                {"mesh": "/Game/Fresh.Head", "output": str(output)}, unreal
            )
            saved = json.loads(output.read_text())
        self.assertEqual(result["vertex_count"], 3)
        self.assertEqual(result["triangle_count"], 2)
        self.assertNotEqual(saved["triangle_uvs"][0], saved["triangle_uvs"][1])
        self.assertTrue(result["native_uvs_preserved"])
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()

    def test_occupied_texture_namespace_cannot_overwrite_an_asset(self):
        unreal = self.fixture()
        unreal.EditorAssetLibrary.list_assets.return_value = ["/Game/Existing"]
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            import_base_color_texture({"destination": "/Game/Existing"}, unreal)
        unreal.AssetToolsHelpers.get_asset_tools.assert_not_called()

    def test_head_in_a_package_directory_cannot_turn_eye_material_into_skin(self):
        unreal = self.fixture()
        material = SimpleNamespace(get_path_name=lambda: "/Game/HeadOnlyDraft/Eye.Eye")
        unreal.SkeletalMesh.get_editor_property = lambda mesh, name: [
            SimpleNamespace(material_slot_name="Head", material_interface=None),
            SimpleNamespace(material_slot_name="Eye", material_interface=material),
        ]
        unreal.GeometryScript_Materials.get_triangle_material_id = lambda mesh, index: (
            index,
            True,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = export_skin_uv_mesh(
                {
                    "mesh": "/Game/Fresh.Head",
                    "output": str(Path(directory) / "uv.json"),
                },
                unreal,
            )
        self.assertEqual(result["triangle_count"], 1)

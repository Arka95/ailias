"""Test UV surface correspondence and photo selection without models or GPU work."""

import unittest
import tempfile
from pathlib import Path

import numpy as np

from ailias.facefitting.texture import choose_texture_views, uv_surface_map, validate_material_textures


class TextureTests(unittest.TestCase):
    def test_material_texture_resolves_locally_not_above_the_export_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texture = root / "skin.png"
            texture.write_bytes(b"synthetic texture reference, never decoded")
            material = root / "head.mtl"
            material.write_text("newmtl skin\nmap_Kd skin.png\n")
            validate_material_textures(material, texture)
            material.write_text("newmtl skin\nmap_Kd ../../skin.png\n")
            with self.assertRaisesRegex(ValueError, "retained skin texture"):
                validate_material_textures(material, texture)

    def test_uv_texel_centers_reconstruct_from_per_corner_barycentric_weights(self):
        uvs = np.array([[[0., 0.], [1., 0.], [0., 1.]]])
        rows, columns, triangles, barycentric = uv_surface_map(uvs, 8)
        actual = np.einsum("ni,nij->nj", barycentric, uvs[triangles])
        expected = np.c_[(columns+.5)/8, 1-(rows+.5)/8]
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_allclose(barycentric.sum(1), 1)

    def test_separate_uv_islands_and_exterior_exclusions_are_preserved(self):
        uvs = np.array([[[0., 0.], [.4, 0.], [0., .4]],
                        [[.6, .6], [1., .6], [.6, 1.]]])
        _, _, triangles, _ = uv_surface_map(uvs, 16)
        self.assertEqual(set(triangles), {0, 1})
        _, _, triangles, _ = uv_surface_map(uvs, 16, [True, False])
        self.assertEqual(set(triangles), {0})
        with self.assertRaisesRegex(ValueError, "No supported"):
            uv_surface_map(uvs, 16, [False, False])

    def test_photo_selection_preserves_other_views_without_filename_rules(self):
        views, entries, observations = [], {}, {}
        for i, category in enumerate(("front", "front", "front", "left_oblique", "right_oblique")):
            face = str(i)
            views.append({"face_id": face, "view": {"category": category},
                          "median_outer_eye_span_error": .01 if category == "front" else .05})
            entries[face] = {"selection_weight": 1}
            observations[face] = {"blendshapes": {}}
        selected = choose_texture_views(views, entries, observations, 3)
        self.assertEqual({view["view"]["category"] for view in selected},
                         {"front", "left_oblique", "right_oblique"})
        self.assertEqual([v["face_id"] for v in selected], ["0", "3", "4"])


if __name__ == "__main__":
    unittest.main()

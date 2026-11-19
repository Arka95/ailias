import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ailias.head_reconstruction.geometry.sculpt_control_mapping import (
    associate_controls,
    map_native_controls,
)
from ailias.storage import Store, atomic_json


class NativeControlMapTests(unittest.TestCase):
    def setUp(self):
        self.vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]
        self.configuration = {
            "landmark_9": {"type": "landmark", "mesh": "head_lod0_mesh", "vID": 0},
            "landmark_01": {"type": "landmark", "mesh": "head_lod0_mesh", "vID": 1},
        }
        self.anatomy = {
            "lip_curve": {"type": "curve", "mesh": "head_lod0_mesh", "vIDs": [0, 1],
                          "start": "mouth_corner"},
            "mouth_corner": {"type": "landmark", "vID": 0},
            "unscoped_point": {"type": "landmark", "vID": 1},
        }

    def associate(self, points=None):
        return associate_controls(points or [self.vertices[1], self.vertices[0]],
                                  self.vertices, self.configuration, self.anatomy)

    def test_api_indices_do_not_assume_configuration_sort_order_or_duplicate_vertex_identity(self):
        result = self.associate()
        self.assertEqual(result["status"], "mapped")
        self.assertEqual([item["vertex_id"] for item in result["controls"]], [1, 0])
        self.assertEqual(result["controls"][1]["configuration_names"], ["landmark_9"])
        self.assertEqual(result["controls"][1]["anatomical_memberships"], ["lip_curve", "mouth_corner"])
        self.assertNotIn("unscoped_point", result["controls"][0]["anatomical_memberships"])
        self.assertFalse(result["automatic_sculpt_proposal"])

    def test_no_nearest_anatomical_guess_for_unmatched_control(self):
        result = self.associate([[0.01, 0, 0]])
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["controls"][0]["reason"], "no_coincident_definition")
        self.assertNotIn("vertex_id", result["controls"][0])

    def test_distinct_coincident_canonical_definitions_are_ambiguous(self):
        self.configuration["other"] = {"type": "landmark", "mesh": "head_lod0_mesh", "vID": 3}
        result = self.associate([self.vertices[0]])
        self.assertEqual(result["controls"][0]["reason"], "ambiguous_coincident_definitions")
        self.assertEqual(result["controls"][0]["candidate_vertex_ids"], [0, 3])

    def test_duplicate_controls_invalid_metadata_and_nonfinite_positions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Multiple native controls"):
            self.associate([self.vertices[0], self.vertices[0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.associate([[float("nan"), 0, 0]])
        self.configuration["landmark_9"]["vID"] = -1
        with self.assertRaisesRegex(ValueError, "vertex IDs"):
            self.associate()

    def test_other_mesh_entries_do_not_override_head_anatomy(self):
        self.configuration["tooth"] = {"type": "landmark", "mesh": "teeth_lod0_mesh", "vID": 0}
        self.anatomy["tooth_curve"] = {"type": "curve", "mesh": "teeth_lod0_mesh", "vIDs": [0]}
        result = self.associate()
        self.assertNotIn("tooth_curve", result["controls"][1]["anatomical_memberships"])
        self.assertEqual(result["controls"][1]["configuration_names"], ["landmark_9"])

    def test_unscoped_endpoint_shared_by_different_meshes_is_not_silently_assigned_to_head(self):
        self.anatomy["tooth_curve"] = {
            "type": "curve", "mesh": "teeth_lod0_mesh", "vIDs": [0], "start": "mouth_corner"}
        self.assertNotIn("mouth_corner", self.associate()["controls"][1]["anatomical_memberships"])

    def test_persistent_map_requires_matching_unchanged_source_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store(root / "data", "test")
            source = root / "head.uasset"
            source.write_bytes(b"unchanged native package")
            topology = {"mesh": "/Game/Diagnostics/Head.Head", "vertices_cm": self.vertices}
            inspection = {"status": "exported", "source_saved": False, "source_unchanged": True,
                          "source_sha256": store.reference_artifact(source)["sha256"],
                          "topology_vertices": len(self.vertices), "landmarks_cm": self.vertices[:2]}
            paths = [root / name for name in ("topology.json", "inspection.json", "config.json", "anatomy.json")]
            for path, data in zip(paths, (topology, inspection, self.configuration, self.anatomy)):
                atomic_json(path, data)
            result = map_native_controls(store, *paths, source)
            self.assertEqual(json.loads(Path(result["manifest"]).read_text())["status"], "mapped")
            self.assertTrue(all(store.verify(item) for item in result["input_dependencies"]))
            source.write_bytes(b"changed package")
            with self.assertRaisesRegex(ValueError, "unchanged source package"):
                map_native_controls(store, *paths, source)
            source.write_bytes(b"unchanged native package")
            def change_source(*args):
                source.write_bytes(b"mid-run change")
                return {}

            with patch("ailias.head_reconstruction.geometry.sculpt_control_mapping.associate_controls", side_effect=change_source):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    map_native_controls(store, *paths, source)

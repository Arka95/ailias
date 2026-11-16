import copy
import math
import unittest

from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.imageprocessing.subjects import associate_subject, select_views
from ailias.imageprocessing.views import POSE_POINTS, estimate_view


def face(name, vector, category="front", source=None, confidence=0.9):
    return {
        "id": name, "source_id": source or name, "embedding": vector,
        "view": {"category": category, "confidence": confidence},
        "quality": {"reasons": [], "native_short_side": 600, "laplacian_variance": 100},
    }


class SubjectAssociationTests(unittest.TestCase):
    def test_links_other_pose_clusters_from_multiple_seed_sources_without_chaining(self):
        faces = [face("a", [1, 0]), face("b", [0.98, 0.1]),
                 face("profile", [0.8, 0.6]), face("bridge", [0.25, 0.97])]
        groups = [{"id": "seed", "members": ["a", "b"]},
                  {"id": "other", "members": ["profile", "bridge"]}]
        before = copy.deepcopy(faces)
        selected, decisions = associate_subject(faces, groups, "seed", ProcessingPolicy())
        self.assertEqual({f["id"] for f in selected}, {"a", "b", "profile"})
        self.assertEqual(next(d for d in decisions if d["face_id"] == "bridge")["decision"], "review")
        self.assertEqual(faces, before, "Association must not rewrite persisted imageprocessing groups")

    def test_rejects_same_photo_faces_and_stronger_cooccurring_identity_competitors(self):
        faces = [face("a", [1, 0], source="photo"), face("b", [1, 0]),
                 face("other", [0.6, 0.8], source="photo"), face("uncertain", [0.6, 0.8])]
        groups = [{"id": "seed", "members": ["a", "b"]},
                  {"id": "other", "members": ["other", "uncertain"]}]
        selected, decisions = associate_subject(faces, groups, "seed", ProcessingPolicy())
        self.assertEqual({f["id"] for f in selected}, {"a", "b"})
        results = {record["face_id"]: record["decision"] for record in decisions}
        self.assertEqual(results["other"], "excluded_cooccurring_face")
        self.assertEqual(results["uncertain"], "review")

    def test_requires_independent_anchor_sources_and_can_disable_expansion(self):
        faces = [face("a", [1, 0]), face("candidate", [1, 0])]
        groups = [{"id": "seed", "members": ["a"]}, {"id": "other", "members": ["candidate"]}]
        selected, _ = associate_subject(faces, groups, "seed", ProcessingPolicy())
        self.assertEqual([f["id"] for f in selected], ["a"])
        selected, decisions = associate_subject(faces, groups, "seed", ProcessingPolicy(), expand=False)
        self.assertEqual([f["id"] for f in selected], ["a"])
        self.assertEqual(decisions[1]["decision"], "not_requested")

    def test_two_associated_faces_in_one_photo_are_abstained_not_arbitrarily_deduplicated(self):
        faces = [face("a", [1, 0]), face("b", [1, 0]),
                 face("c", [1, 0], source="shared"), face("d", [1, 0], source="shared")]
        groups = [{"id": "seed", "members": ["a", "b"]}, {"id": "other", "members": ["c", "d"]}]
        selected, decisions = associate_subject(faces, groups, "seed", ProcessingPolicy())
        self.assertEqual({f["id"] for f in selected}, {"a", "b"})
        self.assertTrue(all(d["decision"] == "review_cooccurrence_conflict"
                            for d in decisions if d["source_id"] == "shared"))


class ViewSelectionTests(unittest.TestCase):
    def test_prioritizes_frontal_anchors_without_losing_complementary_angles(self):
        candidates = [face(f"front-{i}", [1, 0]) for i in range(60)]
        for category in ("left_oblique", "right_oblique", "left_profile", "right_profile"):
            candidates.extend(face(f"{category}-{i}", [1, 0], category) for i in range(15))
        chosen, counts = select_views(candidates, 40)
        self.assertEqual(len(chosen), 40)
        self.assertEqual(chosen[0]["view"]["category"], "front")
        self.assertGreaterEqual(counts["front"], 20)
        self.assertTrue(all(counts[name] >= 1 for name in
                            ("left_oblique", "right_oblique", "left_profile", "right_profile")))

    def test_keeps_unknown_pose_available_and_prefers_clean_fronts_over_warning_fronts(self):
        warning = face("warning", [1, 0])
        warning["quality"]["reasons"] = ["face_underexposed"]
        candidates = [warning, face("clean", [1, 0]), face("unknown", [1, 0], "unknown")]
        chosen, _ = select_views(candidates, 2)
        self.assertEqual([f["id"] for f in chosen], ["clean", "unknown"])

    def test_retains_previously_tracked_backbone_within_the_same_view_budget(self):
        candidates = [face(f"new-{i}", [1, 0]) for i in range(20)]
        candidates.append(face("tracked", [1, 0], "unknown", confidence=0.1))
        chosen, counts = select_views(candidates, 8, ["tracked"])
        self.assertIn("tracked", {item["id"] for item in chosen})
        self.assertEqual(len(chosen), 8)
        self.assertGreater(counts["front"], counts["unknown"])

    def test_canonical_projections_classify_front_and_both_oblique_sides(self):
        import cv2
        import numpy as np

        model = np.asarray(POSE_POINTS, dtype=np.float64)
        intrinsics = np.array([[1200., 0, 500], [0, 1200, 500], [0, 0, 1]])
        for angle, expected in ((0, "front"), (-40, "left_oblique"), (40, "right_oblique")):
            with self.subTest(angle=angle):
                projected = cv2.projectPoints(
                    model, np.array([0., math.radians(angle), 0.]),
                    np.array([0., 0., 80.]), intrinsics, None)[0].reshape(5, 2)
                result = estimate_view({"landmarks_5": projected.tolist()}, {"oriented_size": [1000, 1000]})
                self.assertEqual(result["category"], expected, result)
                self.assertFalse(result["calibrated"])

    def test_missing_or_degenerate_landmarks_abstain_without_inventing_angles(self):
        for landmarks in ([], [[1, 1]]*5):
            result = estimate_view({"landmarks_5": landmarks}, {"oriented_size": [1000, 1000]})
            self.assertEqual(result["category"], "unknown")
            self.assertNotIn("yaw_deg", result)


if __name__ == "__main__":
    unittest.main()

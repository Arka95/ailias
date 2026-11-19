"""Typed fitting exports and retained-input checks without neural/native execution."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ailias.head_reconstruction.pipeline import fit_head_from_photos
from ailias.head_reconstruction.geometry.tests.fitting_fixtures import make_head_fit
from ailias.head_reconstruction.geometry.photo_camera import prepare_photo_tracking
from ailias.head_reconstruction.settings import FitPolicy
from ailias.storage import Store, atomic_json


MODULE = "ailias.head_reconstruction.pipeline"


class PhotoHeadJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name), "subject")
        self.store.root.mkdir()
        self.source = self.store.root / "source.png"
        Image.new("RGBA", (120, 100), (100, 80, 60, 255)).save(self.source)
        self.reference = self.store.reference_artifact(self.source)
        alpha = self.store.root / "alpha.png"
        Image.new("L", (120, 100), 255).save(alpha)
        alpha_reference = self.store.reference_artifact(alpha)
        alignment = {
            "status": "ready",
            "source_to_aligned": np.eye(3).tolist(),
            "aligned_to_source": np.eye(3).tolist(),
        }
        self.ids = [f"view-{index}" for index in range(5)]
        entries = [
            {
                "face_id": view_id,
                "source_id": self.reference["sha256"],
                "source": self.reference,
                "portrait_bounds": [0, 0, 120, 100],
                "foreground_alpha": alpha_reference,
                "analysis": {"aligned_cutout": self.reference, "alignment": alignment},
            }
            for view_id in self.ids
        ]
        self.inputs = self.store.root / "sources.json"
        atomic_json(
            self.inputs,
            {
                "inputs": {"entries": entries, "curation_sha256": "reviewed"},
            },
        )
        self.prepared = {
            "manifest": "sources.json",
            "inputs": {"entries": entries, "curation_sha256": "reviewed"},
        }
        self.extraction = {
            "sources": {
                self.reference["sha256"]: {
                    "metadata": {"oriented_size": [120, 100]},
                }
            },
            "backend": {},
            "policy": {},
        }
        self.observations = {
            view_id: {
                "face_id": view_id,
                "source_id": self.reference["sha256"],
                "status": "ready",
                "source_size": [120, 100],
                "aligned_cutout": self.reference,
                "alignment": alignment,
                "aligned_xy": np.tile([60.0, 50.0], (478, 1)).tolist(),
                "source_xy": np.tile([60.0, 50.0], (478, 1)).tolist(),
                "on_head": [True] * 478,
                "eye_span_source": 30.0,
            }
            for view_id in self.ids
        }
        self.views = [
            {
                "face_id": view_id,
                "source_size": np.array([120, 100]),
                "weights": np.ones(468),
                "view": {"category": "front"},
            }
            for view_id in self.ids
        ]
        vertices = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0.1]])
        self.model = SimpleNamespace(
            vertices=vertices,
            triangles=np.array([[0, 1, 2]]),
            center=np.zeros(3),
            expression_names=["jaw"],
            identity_names=["shape"],
            neutral=lambda identity: vertices + identity[0] * 0.01,
            audit=lambda: {"source": "synthetic"},
        )
        self.fitted = make_head_fit(self.ids, identity=np.array([0.25]))
        self.detector = Mock()
        self.policy = FitPolicy(
            device="cpu", identity_candidates=(1,), expression_modes=(1, 1, 1)
        )

    def run_job(self, *, changed_source=False, insufficient_views=False):
        def prepare(model, observations, entries, *, report, **kwargs):
            report.update(prepared_view_ids=self.ids, excluded=[])
            if insufficient_views:
                report.update(
                    prepared_view_ids=self.ids[:1],
                    excluded=[
                        {"face_id": self.ids[1], "reason": "insufficient_support"}
                    ],
                )
                raise ValueError("Need at least five reliable dense views")
            return self.views, []

        def fit(model, views, policy, report, *, on_candidate):
            report.inputs.training_view_ids = self.ids[:-1]
            report.inputs.validation_view_ids = self.ids[-1:]
            on_candidate(1, self.fitted)
            return self.fitted

        def render(*args):
            if changed_source:
                self.source.write_bytes(b"changed after fitting")
            return {}, {"source": "mocked renderer"}

        assets = {
            name: {"artifact": self.reference}
            for name in ("gnm", "correspondence", "landmarker")
        }
        with ExitStack() as stack:
            for name, value in {
                "prepare_inputs": lambda *args: self.prepared,
                "model_assets": lambda *args: assets,
                "load_run": lambda *args: self.extraction,
                "HeadModel": lambda *args: self.model,
                "DenseDetector": lambda *args: self.detector,
                "observe": lambda store, entry, *args: self.observations[
                    entry["face_id"]
                ],
                "prepare_views": prepare,
                "fit_shared_head_identity": fit,
                "render_clay_portrait": render,
                "load_filter": lambda *args: SimpleNamespace(fingerprint="reviewed"),
            }.items():
                stack.enter_context(patch(MODULE + "." + name, side_effect=value))
            return fit_head_from_photos(
                self.store,
                "intake",
                "front",
                self.policy,
                SimpleNamespace(),
                object(),
            )

    def test_typed_fit_exports_existing_camera_and_parameter_contracts(self):
        result = self.run_job()
        self.assertEqual(result["status"], "awaiting_human_likeness_review")
        self.assertEqual(result["schema_version"], 2)
        cameras = json.loads(
            self.store.artifact_path(result["artifacts"]["cameras"]).read_text()
        )
        self.assertEqual([view["face_id"] for view in cameras["views"]], self.ids)
        self.assertEqual(cameras["views"][0]["source_size"], [120, 100])
        self.assertEqual(cameras["views"][0]["view"]["category"], "front")
        with np.load(
            self.store.artifact_path(result["artifacts"]["parameters"])
        ) as data:
            np.testing.assert_array_equal(data["identity"], [0.25])
            self.assertEqual(data["per_view"].shape, (5, 8))
        self.assertTrue(self.store.verify(result["artifacts"]["candidate_1"]))
        self.assertTrue(self.store.verify(result["artifacts"]["landmark_preview"]))
        self.assertEqual(
            result["reconstruction"]["inputs"]["training_view_ids"], self.ids[:-1]
        )
        tracking = self.store.root / "tracking"
        prepare_photo_tracking(self.store, result["job"], self.ids[0], tracking)
        with Image.open(tracking / "tracking-view.png") as image:
            self.assertEqual(image.size, (120, 100))
        tracked = json.loads((tracking / "tracking-camera.json").read_text())
        self.assertEqual(tracked["source_size"], [120, 100])
        self.assertEqual(tracked["reference_face_id"], self.ids[0])
        self.detector.close.assert_called_once()

    def test_source_change_after_fitting_cannot_publish_a_reviewable_head(self):
        result = self.run_job(changed_source=True)
        self.assertEqual(result["status"], "failed")
        self.assertIn("input changed", result["error"]["message"])
        self.assertEqual(result["reconstruction"]["status"], "failed")

    def test_insufficient_views_persist_exclusions_without_exporting_a_mesh(self):
        result = self.run_job(insufficient_views=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reconstruction"]["prepared_view_ids"], self.ids[:1])
        self.assertEqual(
            result["reconstruction"]["excluded"],
            [
                {
                    "view_id": self.ids[1],
                    "reason": "insufficient_support",
                }
            ],
        )
        self.assertNotIn("mesh", result["artifacts"])
        self.detector.close.assert_called_once()

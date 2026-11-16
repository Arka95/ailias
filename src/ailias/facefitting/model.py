import json
from pathlib import Path

import numpy as np

from ailias.facefitting.settings import MODEL_SPECS
from ailias.imageprocessing.models import download
from ailias.storage import atomic_json, digest


def model_assets(store, install=False):
    specs = json.loads(MODEL_SPECS.read_text(encoding="utf-8"))
    result = {}
    for name, spec in specs.items():
        path = store.resolve(str(Path("facefitting") / "models" / spec["file"]))
        if not path.is_file():
            if not install:
                raise ValueError("Missing head-fitting assets; run facefitting models first")
            download(spec["url"], path, spec["limit"], checksum=spec["sha256"])
        if digest(path) != spec["sha256"]:
            raise ValueError(f"Head-fitting model/provenance changed: {name}")
        result[name] = {**spec, "artifact": store.reference_artifact(path)}
    if install:
        atomic_json(store.resolve(str(Path("facefitting") / "models" / "manifest.json")), result)
    return result


def correspondence(table, vertices, triangles, exterior):
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 6 or not np.isfinite(table).all():
        raise ValueError("Correspondence requires finite alternating vertex/weight triples")
    raw_indices, weights = table[:, ::2], table[:, 1::2]
    if (np.any(raw_indices != np.floor(raw_indices)) or np.any(raw_indices < 0)
            or np.any(raw_indices >= len(vertices)) or np.any(weights < 0)
            or np.any(abs(weights.sum(1) - 1) > 2e-6)):
        raise ValueError("Invalid barycentric indices or weights")
    indices = raw_indices.astype(int)
    topology = {tuple(sorted(t)) for t in triangles.tolist()}
    if any(tuple(sorted(t)) not in topology for t in indices):
        raise ValueError("Correspondence does not lie on model triangles")
    weights = weights / weights.sum(1, keepdims=True)
    supported = np.all((exterior[indices] > 0.5) | (weights <= 1e-5), axis=1)
    return indices, weights, supported


def sample_vertices(values, indices, weights):
    return np.sum(values[..., indices, :] * weights[..., None], axis=-2)


def vertex_normals(vertices, triangles):
    normals = np.zeros_like(vertices)
    faces = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                     vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], faces)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


class HeadModel:
    def __init__(self, path, mapping, expression_modes):
        with np.load(path, allow_pickle=False) as data:
            if str(data["version"]) != "3.0" or str(data["variant"]) != "head":
                raise ValueError("Require pinned GNM 3.0 head topology")
            self.vertices = data["template_vertex_positions"].astype(float)
            self.triangles = data["triangles"].astype(int)
            self.identity = data["vertex_identity_basis"][:170].astype(float)
            self.identity_names = data["identity_names"][:170].tolist()
            expression_indices = [i for start, count in zip((0, 100, 200), expression_modes)
                                  for i in range(start, start + count)]
            self.expression = data["expression_basis"][expression_indices].astype(float)
            self.expression_names = data["expression_names"][expression_indices].tolist()
            self.groups = dict(zip(data["vertex_group_names"].tolist(), data["vertex_groups"]))
        self.indices, self.weights, self.supported = correspondence(
            np.loadtxt(mapping), self.vertices, self.triangles, self.groups["skin_exterior"])
        if len(self.indices) != 468:
            raise ValueError("Require 468 MediaPipe surface correspondences")
        self.points = self.sample(self.vertices)
        self.identity_points = self.sample(self.identity)
        self.expression_points = self.sample(self.expression)
        self.normals = self.sample(vertex_normals(self.vertices, self.triangles))
        self.normals /= np.maximum(np.linalg.norm(self.normals, axis=1, keepdims=True), 1e-12)
        self.center = self.points.mean(0)
        regions = {name: np.sum(values[self.indices]*self.weights, axis=1)
                   for name, values in self.groups.items() if name.endswith("_region")}
        self.region_names = list(regions)
        self.regions = np.asarray(list(regions.values())).argmax(axis=0)

    def sample(self, values):
        return sample_vertices(values, self.indices, self.weights)

    def neutral(self, coefficients):
        return self.vertices + np.einsum("i,ivc->vc", coefficients, self.identity[:len(coefficients)])

    def audit(self):
        return {
            "vertices": len(self.vertices), "triangles": len(self.triangles),
            "correspondences": len(self.points),
            "supported_exterior_correspondences": int(self.supported.sum()),
            "excluded_non_exterior_landmark_indices": np.flatnonzero(~self.supported).tolist(),
            "basis": "GNM bind-pose template + identity + expression; joint rotations fixed at zero",
            "metric_scale_calibrated": False,
            "prior_only": ["rear scalp", "ears", "neck", "eyeballs", "teeth", "tongue"],
            "density_is_not_measurement_count": True,
        }

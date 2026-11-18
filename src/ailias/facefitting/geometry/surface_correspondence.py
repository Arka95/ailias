"""Associate rendered landmarks with the same native mesh by visible ray hits."""

import numpy as np


def camera_axes(camera):
    vectors = np.asarray([camera[key] for key in ("position", "right", "up", "forward")], dtype=float)
    size = np.asarray(camera["image_size"], dtype=float)
    focal = camera["focal_pixels"]
    if (vectors.shape != (4, 3) or size.shape != (2,)
            or not np.isfinite(vectors).all() or not np.isfinite(size).all()
            or np.any(size <= 0) or isinstance(focal, bool)
            or not np.isfinite(focal) or focal <= 0
            or not np.allclose(vectors[1:] @ vectors[1:].T, np.eye(3), atol=1e-6)):
        raise ValueError("Native correspondence requires finite orthonormal camera axes and dimensions")
    return vectors, size, float(focal)


def project_native(vertices, camera):
    vectors, size, focal = camera_axes(camera)
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Native projection requires finite Nx3 points")
    relative = points-vectors[0]
    depth = relative @ vectors[3]
    if np.any(depth <= 0):
        raise ValueError("Native projection points must be in front of the camera")
    return np.c_[relative @ vectors[1], -relative @ vectors[2]]*focal/depth[:, None]+size/2


def ray_correspondences(vertices, triangles, landmarks_xy, camera, *, front_face_sign=1):
    """Keep canonical triangles/barycentrics; do not wrap or deform another mesh."""
    surface, topology = np.asarray(vertices, dtype=float), np.asarray(triangles)
    points = np.asarray(landmarks_xy, dtype=float)
    vectors, size, focal = camera_axes(camera)
    if (surface.ndim != 2 or surface.shape[1] != 3 or not len(surface)
            or not np.isfinite(surface).all()
            or topology.ndim != 2 or topology.shape[1] != 3 or not len(topology)
            or not np.issubdtype(topology.dtype, np.integer)
            or np.any(topology < 0) or np.any(topology >= len(surface))
            or points.ndim != 2 or points.shape[1] != 2
            or not np.isfinite(points).all() or front_face_sign not in (-1, 1)):
        raise ValueError("Native ray correspondence requires valid canonical geometry and pixel landmarks")
    faces = surface[topology]
    edge1, edge2 = faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]
    origin = vectors[0]-faces[:, 0]
    cross_origin = np.cross(origin, edge1)
    inside = ((points >= 0) & (points < size)).all(1)
    indices = np.full(len(points), -1, dtype=int)
    barycentric = np.zeros((len(points), 3), dtype=float)
    distances = np.zeros(len(points), dtype=float)
    for index in np.flatnonzero(inside):
        offset = (points[index]-size/2)/focal
        direction = vectors[3]+offset[0]*vectors[1]-offset[1]*vectors[2]
        direction /= np.linalg.norm(direction)
        cross = np.cross(direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, cross)
        supported = determinant*front_face_sign > 1e-10
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=supported)
        u = np.einsum("ij,ij->i", origin, cross)*inverse
        v = (cross_origin @ direction)*inverse
        depth = np.einsum("ij,ij->i", edge2, cross_origin)*inverse
        supported &= ((u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8) & (depth > 1e-8))
        if not supported.any():
            continue
        hit = int(np.argmin(np.where(supported, depth, np.inf)))
        indices[index] = hit
        barycentric[index] = [1-u[hit]-v[hit], u[hit], v[hit]]
        distances[index] = depth[hit]
    return {
        "triangle_indices": indices, "barycentric": barycentric,
        "supported": indices >= 0, "distance_cm": distances,
        "method": "same-native-visible-ray-barycentric-v1",
        "semantic_accuracy_verified": False,
    }


def sample_native(vertices, triangles, correspondence):
    """Return only supported sampled points and their original landmark indices."""
    supported = np.asarray(correspondence["supported"], dtype=bool)
    indices = np.asarray(correspondence["triangle_indices"])
    weights = np.asarray(correspondence["barycentric"])
    surface, topology = np.asarray(vertices), np.asarray(triangles)
    if (indices.shape != supported.shape or weights.shape != (len(indices), 3)
            or not np.isfinite(weights).all()
            or np.any(indices[supported] < 0) or np.any(indices[supported] >= len(topology))
            or np.any(weights[supported] < -1e-8)
            or not np.allclose(weights[supported].sum(1), 1)):
        raise ValueError("Invalid native barycentric correspondence")
    samples = np.sum(surface[topology[indices[supported]]] * weights[supported, :, None], axis=1)
    return samples, np.flatnonzero(supported)


def rebind_native_faces(source_triangles, target_triangles, correspondence):
    """Reorder exact canonical faces; abstain where native triangulation changed."""
    source, target = np.asarray(source_triangles), np.asarray(target_triangles)
    if any(value.ndim != 2 or value.shape[1] != 3
           or not np.issubdtype(value.dtype, np.integer) or np.any(value < 0)
           for value in (source, target)):
        raise ValueError("Rebinding requires canonical native triangle vertex IDs")
    by_vertices = {}
    for index, face in enumerate(target):
        by_vertices.setdefault(tuple(sorted(face)), []).append(index)
    old_indices = np.asarray(correspondence["triangle_indices"])
    old_weights = np.asarray(correspondence["barycentric"])
    old_supported = np.asarray(correspondence["supported"], dtype=bool)
    if (old_indices.shape != old_supported.shape or old_weights.shape != (len(old_indices), 3)
            or np.any(old_indices[old_supported] < 0)
            or np.any(old_indices[old_supported] >= len(source))):
        raise ValueError("Invalid source native face correspondence")
    supported = np.zeros(len(old_indices), dtype=bool)
    indices = np.full(len(old_indices), -1, dtype=int)
    weights = np.zeros_like(old_weights)
    for index in np.flatnonzero(old_supported):
        vertices = source[old_indices[index]]
        candidates = by_vertices.get(tuple(sorted(vertices)), [])
        if len(candidates) != 1:
            continue
        match = candidates[0]
        permutation = [int(np.flatnonzero(vertices == vertex)[0]) for vertex in target[match]]
        indices[index], supported[index] = match, True
        weights[index] = old_weights[index, permutation]
    return {
        **{key: value for key, value in correspondence.items() if key != "distance_cm"},
        "triangle_indices": indices, "barycentric": weights,
        "supported": supported, "native_faces_rebound_by_exact_vertex_ids": True,
        "unmatched_landmark_indices": np.flatnonzero(old_supported & ~supported).tolist(),
    }

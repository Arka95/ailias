"""Project fresh sampled-photo colour into genuine native UVs without changing head geometry."""

import json
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from ailias.facefitting.appearance.uv_rasterization import choose_texture_views, uv_surface_map
from ailias.facefitting.geometry.head_model import vertex_normals
from ailias.storage import atomic_json


def rasterize_unreal_uvs(uvs, size):
    """Unreal V increases down the image; the shared OBJ rasterizer expects it to increase up."""
    raster_uvs = np.clip(np.asarray(uvs, dtype=float), 0, 1).copy()
    raster_uvs[..., 1] = 1-raster_uvs[..., 1]
    return uv_surface_map(raster_uvs, size)


def estimate_photo_alignment(created):
    """Fit the inverse of native conform finalization; retain nonrigid residuals as projection uncertainty."""
    target = np.asarray(created["face_landmarks_solved_in_target_pose"], dtype=float)
    canonical = np.asarray(created["face_landmarks_after"], dtype=float)
    if (target.ndim != 2 or target.shape[1:] != (3,) or len(target) < 3
            or canonical.shape != target.shape or not np.isfinite(target).all()
            or not np.isfinite(canonical).all()):
        raise ValueError("Photo projection requires matching finite pre/post-finalization landmarks")
    target_center, canonical_center = target.mean(0), canonical.mean(0)
    u, _, vt = np.linalg.svd((target-target_center).T @ (canonical-canonical_center))
    if np.linalg.det(vt.T @ u.T) < 0:
        vt[-1] *= -1
    rotation = vt.T @ u.T
    translation = canonical_center-target_center @ rotation.T
    residual = np.linalg.norm(target @ rotation.T+translation-canonical, axis=1)
    return {"rotation": rotation.T.tolist(), "translation_cm": (-translation @ rotation).tolist(),
            "method": "rigid least-squares inverse of native target-pose to A-pose finalization",
            "nonrigid_residual_cm_p50_p95_max": np.quantile(residual, [.5, .95, 1]).tolist()}


class SkinProjectionMesh:
    """Keep the saved native rest shape; approximate photo-expression offsets from the fitted prior."""

    def __init__(self, surface, model, identity):
        vertices = np.asarray(surface["vertices"], dtype=float)
        triangles = np.asarray(surface["triangles"])
        self.uvs = np.asarray(surface["triangle_uvs"], dtype=float)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all() or triangles.ndim != 2
                or triangles.shape[1] != 3 or not np.issubdtype(triangles.dtype, np.integer)
                or not len(triangles) or np.any(triangles < 0) or np.any(triangles >= len(vertices))
                or self.uvs.shape != (len(triangles), 3, 2) or not np.isfinite(self.uvs).all()
                or np.any(self.uvs < -1e-6) or np.any(self.uvs > 1+1e-6)):
            raise ValueError("Native photo projection requires finite skin geometry and matching native UVs")
        alignment = surface.get("canonical_to_target")
        if alignment is not None:
            rotation = np.asarray(alignment["rotation"], dtype=float)
            translation = np.asarray(alignment["translation_cm"], dtype=float)
            if (rotation.shape != (3, 3) or translation.shape != (3,)
                    or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                    or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
                raise ValueError("Native projection alignment must be a finite rigid transform")
            vertices = vertices @ rotation.T + translation
        self.vertices = vertices[:, [0, 2, 1]]/100
        self.triangles = triangles
        self.center = model.center
        distances, indices = cKDTree(model.neutral(identity)).query(self.vertices, k=3)
        weights = 1/np.maximum(distances, 1e-8)
        weights /= weights.sum(1, keepdims=True)
        self.expression = np.einsum("kvpc,vp->kvc", model.expression[:, indices], weights)
        self.groups = {"skin_exterior": np.ones(len(vertices))}
        self.transfer_distance = distances[:, 0]
        source_normals = vertex_normals(model.neutral(identity), model.triangles)
        expected = np.einsum("vpc,vp->vc", source_normals[indices], weights)
        normals = vertex_normals(self.vertices, self.triangles)
        self.winding_reversed_for_projection = np.median((normals*expected).sum(1)) < 0
        if self.winding_reversed_for_projection:
            self.triangles = self.triangles[:, [0, 2, 1]]
            self.uvs = self.uvs[:, [0, 2, 1]]

    def neutral(self, identity):
        return self.vertices


def bake_photo_skin_texture(store, root, surface_path, model, fitted, views, evidence, device,
                       *, texture_size=2048, maximum_views=16, photo_size=1536):
    """Retain coverage and source IDs; unseen texels use labelled median-skin fill, not invented detail."""
    from ailias.facefitting.appearance.cuda_texture_projection import bake

    surface = SkinProjectionMesh(json.loads(surface_path.read_text(encoding="utf-8")), model, fitted["identity"])
    frames = {frame["face_id"]: frame for frame in evidence["frames"]}
    metrics = {view["face_id"]: view for view in fitted["report"]["views"]}
    cameras = [
        {**metrics[view["face_id"]], "face_id": view["face_id"], "view": view["view"],
         "source_size": frames[view["face_id"]]["source_size"],
         "camera": fitted["per_view"][index, :7].tolist(),
         "expression": fitted["per_view"][index, 7:].tolist()}
        for index, view in enumerate(views)
    ]
    entries = {
        face_id: {"portrait_bounds": [0, 0, *frame["source_size"]]}
        for face_id, frame in frames.items() if "source_size" in frame
    }
    selected = choose_texture_views(cameras, entries, frames, maximum_views)
    photos, temporary = [], []
    try:
        for index, view in enumerate(selected):
            face_id = view["face_id"]
            frame, record = frames[face_id], evidence["faces"][face_id]
            with Image.open(store.artifact_path(frame["image"])) as opened:
                image = opened.convert("RGB")
            with Image.open(store.artifact_path(record["regions"])) as opened:
                regions = opened.convert("L")
            source_size = image.size
            image.thumbnail((photo_size, photo_size), Image.Resampling.LANCZOS)
            regions = regions.resize(image.size, Image.Resampling.NEAREST)
            labels, rgb = np.asarray(regions), np.asarray(image)
            skin = record["features"]["regions"]["skin"]
            if skin["status"] != "observed":
                raise ValueError("A selected native texture view lacks skin evidence: " + face_id)
            color = np.asarray(skin["median_linear_rgb"])
            color = np.where(color <= .0031308, color*12.92, 1.055*color**(1/2.4)-.055)
            hair = labels == 17
            data = root / f"native-photo-{index:02}.npz"
            temporary.append(data)
            np.savez_compressed(data, rgb=rgb, labels=labels, alpha=np.uint8(labels != 0)*255)
            photos.append({
                "face_id": face_id, "data": str(data),
                "portrait_scale": (np.asarray(image.size)/source_size).tolist(),
                "skin_color": color.tolist(),
                "hair_color": (np.median(rgb[hair], axis=0)/255).tolist() if hair.sum() >= 32 else None,
                "hair_top_pixel": float(np.quantile(np.nonzero(hair)[0], .02)) if hair.any() else None,
            })
            image.close()
            regions.close()
        mapping = rasterize_unreal_uvs(surface.uvs, texture_size)
        baked = bake(surface, np.zeros(0), mapping, selected, entries, photos, device)
        rows, columns, _, _ = mapping
        atlas = np.broadcast_to(baked["skin_color"], (texture_size, texture_size, 3)).copy()
        coverage = np.zeros((texture_size, texture_size), dtype=bool)
        atlas[rows, columns] = np.where(
            baked["coverage"][:, None], baked["color"], baked["skin_color"])
        coverage[rows, columns] = baked["coverage"]
        if not coverage.any():
            raise RuntimeError("No photographed colour survived native-UV visibility checks")
        distance, nearest = distance_transform_edt(~coverage, return_indices=True)
        padding = (~coverage) & (distance <= 8)
        atlas[padding] = atlas[nearest[0][padding], nearest[1][padding]]
        texture = root / "native-photo-basecolor.png"
        Image.fromarray(np.uint8(np.clip(atlas, 0, 1)*255)).save(texture)
        Image.fromarray(np.uint8(coverage)*255).save(root / "native-photo-coverage.png")
        result = {
            "status": "native_uv_photo_colour_baked", "texture": str(texture),
            "native_uvs_preserved": True, "geometry_modified": False,
            "texture_uv_origin": "top_left; Unreal V increases down image rows",
            "winding_reversed_for_projection": bool(surface.winding_reversed_for_projection),
            "intrinsic_albedo": False, "selected_views": [view["face_id"] for view in selected],
            "projection": baked["report"],
            "expression_transfer": "inverse-distance nearest fitted-GNM vertices; approximate offsets only",
            "expression_transfer_distance_model_units_p50_p95": np.quantile(
                surface.transfer_distance, [.5, .95]).tolist(),
            "unseen_texels": "median observed skin colour, not reconstructed detail",
            "hair_color_srgb": baked["hair_color"].tolist(),
            "hair_length_model_units": baked["hair_length"],
        }
        atomic_json(root / "native-photo-colour.json", result)
        return result
    finally:
        primary = sys.exception()
        for path in temporary:
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                if primary is None:
                    raise
                primary.add_note("Temporary photo cleanup failed: " + type(error).__name__)

"""Project current photo evidence onto a supplied head surface and write its atlas."""

from contextlib import ExitStack
from collections.abc import Mapping
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from ailias.head_reconstruction.appearance.uv_rasterization import choose_texture_views
from ailias.head_reconstruction.reconstruction_data import HeadFittingResult
from ailias.head_reconstruction.appearance.projection_data import PhotoProjectionInput
from ailias.imageprocessing.image_io import read_oriented_image as read_image


def project_photo_evidence(
    root,
    surface,
    identity,
    mapping,
    fitted: HeadFittingResult,
    views,
    photos_by_view: Mapping[str, PhotoProjectionInput],
    device,
    *,
    maximum_views=16,
    photo_size=1536,
):
    from ailias.head_reconstruction.appearance.cuda_texture_projection import bake

    metrics = {view.view_id: view for view in fitted.summary.views}
    cameras = [
        {
            **metrics[view["face_id"]].to_dict(),
            "face_id": view["face_id"],
            "view": view["view"],
            "source_size": list(photos_by_view[view["face_id"]].source_size),
            "camera": fitted.parameters.cameras_and_expressions[index, :7].tolist(),
            "expression": fitted.parameters.cameras_and_expressions[index, 7:].tolist(),
        }
        for index, view in enumerate(views)
    ]
    entries = {
        view_id: {"portrait_bounds": [0, 0, *photo.source_size]}
        for view_id, photo in photos_by_view.items()
    }
    observations = {
        view_id: {"blendshapes": photo.blendshapes}
        for view_id, photo in photos_by_view.items()
    }
    selected = choose_texture_views(cameras, entries, observations, maximum_views)
    photos, temporary = [], []
    try:
        for index, view in enumerate(selected):
            face_id = view["face_id"]
            photo = photos_by_view[face_id]
            with ExitStack() as resources:
                image = resources.enter_context(read_image(photo.image)[0])
                opened_regions = resources.enter_context(Image.open(photo.regions))
                regions = resources.enter_context(opened_regions.convert("L"))
                opened_head = resources.enter_context(Image.open(photo.head_image))
                alpha = resources.enter_context(opened_head.getchannel("A"))
                if (
                    image.size != photo.source_size
                    or regions.size != image.size
                    or alpha.size != image.size
                ):
                    raise ValueError(
                        "Photo pixels, parser labels and head alpha must share the fitted grid"
                    )
                source_size = image.size
                image.thumbnail((photo_size, photo_size), Image.Resampling.LANCZOS)
                regions = resources.enter_context(
                    regions.resize(image.size, Image.Resampling.NEAREST)
                )
                alpha = resources.enter_context(
                    alpha.resize(image.size, Image.Resampling.LANCZOS)
                )
                labels, rgb = np.asarray(regions), np.asarray(image)
                if photo.skin_linear_rgb is None:
                    raise ValueError(
                        "A selected texture view lacks skin evidence: " + face_id
                    )
                color = np.asarray(photo.skin_linear_rgb)
                color = np.where(
                    color <= 0.0031308,
                    color * 12.92,
                    1.055 * color ** (1 / 2.4) - 0.055,
                )
                hair = labels == 17
                data = root / f"photo-texture-{index:02}.npz"
                temporary.append(data)
                np.savez_compressed(
                    data, rgb=rgb, labels=labels, alpha=np.asarray(alpha)
                )
                photos.append(
                    {
                        "face_id": face_id,
                        "data": str(data),
                        "portrait_scale": (
                            np.asarray(image.size) / source_size
                        ).tolist(),
                        "skin_color": color.tolist(),
                        "hair_color": (np.median(rgb[hair], axis=0) / 255).tolist()
                        if hair.sum() >= 32
                        else None,
                        "hair_top_pixel": float(np.quantile(np.nonzero(hair)[0], 0.02))
                        if hair.any()
                        else None,
                    }
                )
        baked = bake(surface, identity, mapping, selected, entries, photos, device)
        return baked, selected
    finally:
        primary = sys.exception()
        for path in temporary:
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                if primary is None:
                    raise
                primary.add_note(
                    "Temporary photo cleanup failed: " + type(error).__name__
                )


def write_texture_atlas(root, mapping, baked, size, *, texture_name, coverage_name):
    rows, columns, _, _ = mapping
    atlas = np.broadcast_to(baked["skin_color"], (size, size, 3)).copy()
    coverage = np.zeros((size, size), dtype=bool)
    atlas[rows, columns] = np.where(
        baked["coverage"][:, None], baked["color"], baked["skin_color"]
    )
    coverage[rows, columns] = baked["coverage"]
    if not coverage.any():
        raise RuntimeError("No photographed colour survived UV visibility checks")
    distance, nearest = distance_transform_edt(~coverage, return_indices=True)
    padding = (~coverage) & (distance <= 8)
    atlas[padding] = atlas[nearest[0][padding], nearest[1][padding]]
    texture = root / texture_name
    mask = root / coverage_name
    Image.fromarray(np.uint8(np.clip(atlas, 0, 1) * 255)).save(texture)
    Image.fromarray(np.uint8(coverage) * 255).save(mask)
    return texture, mask

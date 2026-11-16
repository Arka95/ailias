"""Compute face-quality evidence, embedding groups, subject crops and review sheets.

Embedding similarity groups candidates; it does not measure facial geometry.
"""

import math
from dataclasses import asdict

from PIL import Image, ImageOps

from ailias.imageprocessing.settings import ProcessingPolicy

GROUPING_METHOD = "agglomerative-complete-link-v2"


def clipped_box(box, size):
    x, y, width, height = box
    if not all(math.isfinite(value) for value in box) or width <= 0 or height <= 0:
        raise ValueError("Invalid detected face box")
    bounds = (max(0, math.floor(round(x, 9))), max(0, math.floor(round(y, 9))),
              min(size[0], math.ceil(round(x + width, 9))),
              min(size[1], math.ceil(round(y + height, 9))))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Detected face is outside the image")
    return bounds


def portrait_box(box, size, policy: ProcessingPolicy):
    x, y, width, height = box
    return clipped_box([
        x - width * policy.portrait_side_margin,
        y - height * policy.portrait_top_margin,
        width * (1 + 2 * policy.portrait_side_margin),
        height * (1 + policy.portrait_top_margin + policy.portrait_bottom_margin),
    ], size)


def face_quality(image, box, policy: ProcessingPolicy):
    import numpy as np

    bounds = clipped_box(box, image.size)
    face = image.crop(bounds).convert("L")
    native_size = min(face.size)
    face.thumbnail((256, 256))
    values = np.asarray(face, dtype=np.float32)
    variance = 0.0
    if min(values.shape) > 2:
        laplacian = (values[:-2, 1:-1] + values[2:, 1:-1]
                     + values[1:-1, :-2] + values[1:-1, 2:]
                     - 4 * values[1:-1, 1:-1])
        variance = float(laplacian.var())
    dark, bright = float((values < 16).mean()), float((values >= 240).mean())
    reasons = []
    if native_size < policy.min_face_size:
        reasons.append("face_too_small")
    if variance < policy.min_laplacian_variance:
        reasons.append("face_blurred_or_low_detail")
    if dark > policy.max_dark_fraction:
        reasons.append("face_underexposed")
    if bright > policy.max_bright_fraction:
        reasons.append("face_overexposed")
    return {"native_short_side": native_size, "laplacian_variance": round(variance, 3),
            "dark_fraction": round(dark, 4), "bright_fraction": round(bright, 4),
            "reasons": reasons}


def normalize_embedding(embedding):
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError("Invalid face embedding")
    norm = math.sqrt(sum(value * value for value in embedding))
    if norm <= 1e-12:
        raise ValueError("Empty face embedding")
    return [float(value / norm) for value in embedding]


def similarity(first, second):
    if len(first) != len(second):
        raise ValueError("Incompatible face embedding dimensions")
    return sum(a * b for a, b in zip(first, second))


def group_faces(faces: list[dict], policy: ProcessingPolicy) -> list[dict]:
    import numpy as np

    if not faces:
        return []
    ordered = sorted(faces, key=lambda face: (
        -face["quality"]["native_short_side"], -face["score"], face["id"],
    ))
    vectors = np.asarray([face["embedding"] for face in ordered], dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Incompatible or invalid face embeddings")
    pairwise = np.clip(vectors @ vectors.T, -1, 1)
    scores = np.round(pairwise, 12)
    sources = np.asarray([face["source_id"] for face in ordered])
    scores[sources[:, None] == sources[None, :]] = -np.inf
    clusters = {index: [index] for index in range(len(ordered))}
    while True:
        first, second = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if scores[first, second] < policy.identity_cosine_threshold:
            break
        # Complete linkage propagates both weak matches and co-occurrence exclusions.
        clusters[first].extend(clusters.pop(second))
        merged = np.minimum(scores[first], scores[second])
        scores[first, :] = merged
        scores[:, first] = merged
        scores[second, :] = -np.inf
        scores[:, second] = -np.inf
        scores[first, first] = -np.inf
    groups = []
    for members in clusters.values():
        for index in members:
            face = ordered[index]
            own = [other for other in members if other != index]
            own_score = float(pairwise[index, own].min()) if own else 1.0
            alternatives = [
                float(pairwise[index, other].min())
                for other in clusters.values()
                if other is not members
                and all(ordered[member]["source_id"] != face["source_id"] for member in other)
            ]
            alternative = max(alternatives, default=-1.0)
            # Flag ambiguity after clustering; ambiguous singletons must not become
            # competing anchors that recursively fragment all subsequent matches.
            face["identity_ambiguous"] = (
                alternative >= policy.identity_cosine_threshold
                and own_score - alternative < policy.identity_ambiguity_margin
            )
        groups.append([ordered[index] for index in sorted(members)])
    groups.sort(key=lambda group: (-len(group), min(face["id"] for face in group)))
    return [
        {"id": f"face-{index:03d}", "members": [face["id"] for face in group],
         "eligible": sum(face["eligible"] and not face["identity_ambiguous"] for face in group),
         "identity_confirmed": False, "policy": {
             key: value for key, value in asdict(policy).items() if key.startswith("identity_")
         }}
        for index, group in enumerate(groups, 1)
    ]


def overlaps_other(portrait, box):
    x, y, width, height = box
    area = max(0, min(portrait[2], x + width) - max(portrait[0], x))
    area *= max(0, min(portrait[3], y + height) - max(portrait[1], y))
    return area / (width * height) > 0.2


def subject_bodies(mask: Image.Image, faces: list) -> list[dict]:
    import cv2
    import numpy as np

    reduced = mask.copy()
    reduced.thumbnail((1024, 1024), Image.Resampling.NEAREST)
    binary = (np.asarray(reduced) > 127).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    sx, sy = reduced.width / mask.width, reduced.height / mask.height
    assignments = []
    for face in faces:
        x, y, width, height = face.box
        patch = labels[max(0, int((y + height * 0.2) * sy)):
                       max(0, math.ceil((y + height * 0.8) * sy)),
                       max(0, int((x + width * 0.2) * sx)):
                       max(0, math.ceil((x + width * 0.8) * sx))]
        counts = np.bincount(patch.flatten())
        if len(counts):
            counts[0] = 0
        label = int(counts.argmax()) if len(counts) and counts.max() > patch.size * 0.2 else 0
        assignments.append(label)
    results = []
    for label in assignments:
        if not label:
            results.append({"status": "no_associated_foreground"})
        elif assignments.count(label) > 1:
            results.append({"status": "ambiguous_touching_subjects"})
        else:
            x, y, width, height, _ = stats[label]
            component = Image.fromarray((labels == label).astype(np.uint8) * 255)
            component = component.resize(mask.size, Image.Resampling.NEAREST)
            bounds = (max(0, math.floor(x / sx)), max(0, math.floor(y / sy)),
                      min(mask.width, math.ceil((x + width) / sx)),
                      min(mask.height, math.ceil((y + height) / sy)))
            results.append({"status": "visible_subject_only", "bounds": bounds, "mask": component})
    return results


def contact_sheet(items: list[tuple[str, Image.Image]], columns: int = 6) -> Image.Image:
    rows = max(1, math.ceil(len(items) / columns))
    sheet = Image.new("RGB", (columns * 200, rows * 230), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        tile = ImageOps.contain(image.convert("RGB"), (190, 195))
        x, y = (index % columns) * 200, (index // columns) * 230
        sheet.paste(tile, (x + 5, y + 5))
        draw.text((x + 5, y + 202), label, fill="black")
    return sheet

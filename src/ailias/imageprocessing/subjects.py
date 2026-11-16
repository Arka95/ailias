"""Seed-anchored association across candidate clusters; never confirms identities."""

import math

from ailias.imageprocessing.features import normalize_embedding, similarity
from ailias.imageprocessing.views import VIEW_WEIGHTS


def associate_subject(faces, groups, seed_id, policy, expand=True):
    by_id = {face["id"]: face for face in faces}
    seed = next(group for group in groups if group["id"] == seed_id)
    anchors = [by_id[key] for key in seed["members"] if key in by_id]
    if not anchors:
        raise ValueError("Seed group has no current human-kept usable portraits")
    anchor_sources = {face["source_id"] for face in anchors}
    vectors = {key: normalize_embedding(face["embedding"]) for key, face in by_id.items()}
    if len({len(vector) for vector in vectors.values()}) != 1:
        raise ValueError("Selection requires one compatible embedding space")
    centroid = normalize_embedding([sum(values) for values in zip(*(vectors[a["id"]] for a in anchors))])
    group_ids = {key: group["id"] for group in groups for key in group["members"]}
    # Co-occurrence gives actual cannot-link evidence; alternative pose clusters do not.
    competitors = [face for face in faces if face["source_id"] in anchor_sources
                   and face["id"] not in seed["members"]]
    selected, decisions = [], []
    for face in faces:
        key, source = face["id"], face["source_id"]
        record = {"face_id": key, "group": group_ids.get(key), "source_id": source,
                  "identity_confirmed": False}
        if key in seed["members"]:
            record.update(decision="seed", support_sources=[], centroid_cosine=None)
        elif not expand:
            record.update(decision="not_requested")
        elif source in anchor_sources:
            record.update(decision="excluded_cooccurring_face")
        else:
            supports = sorted(
                ((similarity(vectors[key], vectors[a["id"]]), a["source_id"]) for a in anchors
                 if a["source_id"] != source), reverse=True)
            unique = {}
            for score, source_id in supports:
                unique.setdefault(source_id, score)
            supporting = [(source_id, score) for source_id, score in unique.items()
                          if score >= policy.identity_cosine_threshold]
            center_score = similarity(vectors[key], centroid)
            alternative = max((similarity(vectors[key], vectors[c["id"]]) for c in competitors),
                              default=-1.)
            second = supporting[1][1] if len(supporting) >= 2 else -1.
            accepted = (
                len(supporting) >= 2
                and center_score >= policy.identity_cosine_threshold - policy.identity_ambiguity_margin
                and second - alternative >= policy.identity_ambiguity_margin
            )
            record.update(decision="associated" if accepted else "review",
                          centroid_cosine=round(center_score, 5),
                          competing_cosine=round(alternative, 5),
                          support_sources=[source_id for source_id, _ in supporting])
        decisions.append(record)
        if record["decision"] in {"seed", "associated"}:
            selected.append({**face, "subject_association": record})
    # Exclude two proposed subject faces from one photo rather than silently selecting one.
    sources = {}
    for face in selected:
        sources.setdefault(face["source_id"], []).append(face)
    conflicts = {key for key, members in sources.items() if len(members) > 1}
    for record in decisions:
        if record["source_id"] in conflicts and record["decision"] in {"seed", "associated"}:
            record["decision"] = "review_cooccurrence_conflict"
    return [face for face in selected if face["source_id"] not in conflicts], decisions


def view_rank(face):
    quality, view = face["quality"], face["view"]
    light = face.get("lighting", {})
    category = view["category"]
    target_yaw = 35 if "oblique" in category else 70 if "profile" in category else 0
    target_pitch = 30 if category in {"up", "down"} else 0
    pose = .5
    if "yaw_deg" in view and "pitch_deg" in view:
        pose = math.exp(-.5*(((abs(view["yaw_deg"]) - target_yaw)/20)**2
                             + ((abs(view["pitch_deg"]) - target_pitch)/20)**2))
    detail = .6*min(quality["native_short_side"]/800, 1)
    detail += .4*min(math.log1p(max(0, quality["laplacian_variance"]))/math.log1p(400), 1)
    score = .45*pose + .35*detail + .2*(1 - light.get("baseline_penalty", 0.))
    return (-len(quality["reasons"]), -len(light.get("warnings", [])), score,
            min(quality["native_short_side"], 1600), view["confidence"], face["id"])


def select_views(faces, maximum, registration_anchors=()):
    buckets = {name: [] for name in VIEW_WEIGHTS}
    for face in faces:
        buckets[face["view"]["category"]].append(face)

    for bucket in buckets.values():
        bucket.sort(key=view_rank, reverse=True)
    selected, counts = [], dict.fromkeys(buckets, 0)
    anchors = set(registration_anchors)
    for name, bucket in buckets.items():
        retained = []
        for face in bucket:
            if face["id"] in anchors and len(selected) < maximum:
                selected.append(face)
                counts[name] += 1
            else:
                retained.append(face)
        buckets[name] = retained
    # Cover complementary observed directions before spending the rest on frontal anchors.
    for name in buckets:
        if buckets[name] and not counts[name] and len(selected) < maximum:
            selected.append(buckets[name].pop(0))
            counts[name] += 1
    while len(selected) < maximum:
        available = [name for name, bucket in buckets.items() if bucket]
        if not available:
            break
        name = min(available, key=lambda category: (counts[category]/VIEW_WEIGHTS[category],
                                                   -VIEW_WEIGHTS[category]))
        selected.append(buckets[name].pop(0))
        counts[name] += 1
    return selected, counts

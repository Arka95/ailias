"""Convert explicit human crop decisions into conservative keep/reject/review filters.

Automatic positive matches do not become new human-approved training exemplars.
"""

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ailias.imageprocessing.features import normalize_embedding, similarity
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.storage import Store, atomic_json, digest

PROFILE = str(Path("imageprocessing") / "curation.json")
METHOD = "reviewed-exemplar-cosine-v1"


class CurationError(ValueError):
    """A reviewed gallery must never silently become an unfiltered scan."""


def embedding_space(backend: dict) -> dict:
    return {"provider": backend.get("provider"), "revision": backend.get("revision"),
            "recognizer": Path(backend.get("models", {}).get("sface", "unspecified")).name}


def crop_key(source_id, bounds):
    return source_id, tuple(bounds)


class PortraitFilter:
    def __init__(self, profile: dict, policy: ProcessingPolicy):
        if (profile.get("schema_version") != 1 or profile.get("method") != METHOD
                or not isinstance(profile.get("examples"), list)):
            raise CurationError("Unsupported or malformed portrait curation profile")
        self.profile, self.policy = profile, policy
        self.examples = profile["examples"]
        self.known = {}
        self.hashes = {}
        dimensions = set()
        for example in self.examples:
            if (example["label"] not in {"accept", "reject"}
                    or not re.fullmatch(r"[a-f0-9]{64}", example["portrait_sha256"])):
                raise CurationError("Invalid reviewed portrait label or hash")
            vector = normalize_embedding(example["embedding"])
            dimensions.add(len(vector))
            example["embedding"] = vector
            key = crop_key(example["source_id"], example["portrait_bounds"])
            if key in self.known and self.known[key]["label"] != example["label"]:
                raise CurationError("Conflicting reviewed labels for the same source crop")
            self.known[key] = example
            self.hashes[example["portrait_sha256"]] = example
        if len(dimensions) != 1 or not any(e["label"] == "accept" for e in self.examples):
            raise CurationError("Curation requires kept portraits with consistent face embeddings")
        self.dimension = dimensions.pop()

    def classify(self, embedding, quality: dict, source_id=None, bounds=None,
                 portrait_sha256=None, held_out_source=None) -> dict:
        vector = normalize_embedding(embedding)
        if len(vector) != self.dimension:
            raise CurationError("Face embedding dimensions differ from reviewed examples")
        known = self.hashes.get(portrait_sha256)
        if known is None and bounds is not None:
            known = self.known.get(crop_key(source_id, bounds))
        if known is not None and held_out_source is None:
            return {"decision": known["label"], "reason": "human_reviewed_crop",
                    "reviewed_portrait_sha256": known["portrait_sha256"]}
        nearest = {}
        for label in ("accept", "reject"):
            candidates = [
                (max(-1.0, min(1.0, similarity(vector, example["embedding"]))),
                 example["portrait_sha256"])
                for example in self.examples
                if example["label"] == label and example["source_id"] != held_out_source
            ]
            nearest[label] = max(candidates) if candidates else None
        positive, negative = nearest["accept"], nearest["reject"]
        result = {
            "decision": "review", "reason": "insufficient_or_conflicting_similarity",
            "positive_cosine": positive[0] if positive else None,
            "negative_cosine": negative[0] if negative else None,
            "nearest_kept": positive[1] if positive else None,
            "nearest_rejected": negative[1] if negative else None,
        }
        if positive is None:
            return result
        p, n = positive[0], negative[0] if negative else -1.0
        if (negative is not None and n >= self.policy.curation_reject_cosine
                and p < self.policy.curation_reject_positive_ceiling
                and n - p >= self.policy.curation_score_margin):
            result.update(decision="reject", reason="strong_rejected_exemplar_match")
        elif (p >= self.policy.curation_accept_cosine
              and p - n >= self.policy.curation_score_margin
              and not quality["reasons"]):
            result.update(decision="accept", reason="kept_exemplar_match")
        elif quality["reasons"]:
            result["reason"] = "quality_review_required"
        return result

    def validate_held_out_sources(self) -> dict:
        counts = {"kept": Counter(), "rejected": Counter()}
        for example in self.examples:
            result = self.classify(example["embedding"], example["quality"],
                                   held_out_source=example["source_id"])
            counts["kept" if example["label"] == "accept" else "rejected"][result["decision"]] += 1
        return {
            "method": "leave-one-source-photo-out; no exact-crop overrides",
            "counts": {label: dict(values) for label, values in counts.items()},
            "warning": "Diagnostic on the reviewed collection, not independent future-data accuracy",
        }


def load_filter(store: Store, backend: dict, policy: ProcessingPolicy):
    path = store.resolve(PROFILE)
    if not path.is_file():
        return None
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("alias") != store.root.name or profile.get("embedding_space") != embedding_space(backend):
        raise CurationError("Curation alias/model differs from this scan; rebuild reviewed features first")
    classifier = PortraitFilter(profile, policy)
    for example in classifier.examples:
        if example["label"] == "accept":
            artifact = example["artifact"]
            if (artifact != {"path": str(Path("imageprocessing") / "assets" / "portraits"
                                        / f"{example['portrait_sha256']}.png"),
                             "sha256": example["portrait_sha256"]}
                    or not store.verify(artifact)):
                raise CurationError("Reviewed kept portrait is missing/changed; run curate after manual review")
    classifier.fingerprint = digest(path)
    return classifier


def capture_curation(store: Store, run: str, policy: ProcessingPolicy,
                     removed_are_rejected: bool = False) -> dict:
    """Explicitly freeze a human-reviewed folder; never infer rejection during scans."""
    from ailias.imageprocessing.pipeline import load_run

    if not removed_are_rejected:
        raise CurationError("Confirm completed manual review with --removed-are-rejected")
    report = load_run(store, run)
    previous_path = store.resolve(PROFILE)
    examples = {}
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if (previous.get("alias") != store.root.name
                or previous.get("embedding_space") != embedding_space(report["backend"])):
            raise CurationError("Cannot merge different face embedding models")
        examples.update((e["portrait_sha256"], e) for e in PortraitFilter(previous, policy).examples)
    skipped = 0
    for face in report["faces"].values():
        relative = face["products"].get("portrait")
        if relative is None:
            continue
        checksum = Path(relative).stem
        if not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise CurationError("Consolidate legacy portraits with cleanup before capturing curation")
        accepted = store.resolve(str(Path("imageprocessing") / "assets" / "portraits" / f"{checksum}.png"))
        if not accepted.exists() and store.resolve(relative).is_file():
            skipped += 1
            continue
        examples[checksum] = {
            "portrait_sha256": checksum, "source_id": face["source_id"],
            "portrait_bounds": face["portrait_bounds"], "embedding": face["embedding"],
            "quality": face["quality"],
        }
    for checksum, example in examples.items():
        relative = str(Path("imageprocessing") / "assets" / "portraits" / f"{checksum}.png")
        path = store.resolve(relative)
        if path.is_file():
            if path.is_symlink() or digest(path) != checksum:
                raise CurationError("Kept portrait was edited or renamed; its content hash no longer matches")
            example.update(label="accept", artifact={"path": relative, "sha256": checksum})
        else:
            example["label"] = "reject"
            example.pop("artifact", None)
    profile = {
        "schema_version": 1, "method": METHOD, "alias": store.root.name,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_run": run,
        "labels": "Explicit human review: kept portraits valid; removed portraits rejected",
        "embedding_space": embedding_space(report["backend"]),
        "examples": [examples[key] for key in sorted(examples)],
        "policy": {key: value for key, value in vars(policy).items() if key.startswith("curation_")},
    }
    classifier = PortraitFilter(profile, policy)
    profile["validation"] = classifier.validate_held_out_sources()
    atomic_json(previous_path, profile)
    counts = Counter(e["label"] for e in classifier.examples)
    return {"profile": str(previous_path), "kept": counts["accept"], "rejected": counts["reject"],
            "skipped_unreviewed": skipped, "validation": profile["validation"]}

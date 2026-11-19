"""Compose imageprocessing APIs into fresh, explicitly exported photo evidence."""

from contextlib import ExitStack
from pathlib import Path
import sys

from PIL import Image

from ailias.common.identifiers import validate_identifier
from ailias.common.json_values import iter_artifact_references
from ailias.imageprocessing.face_feature_extraction import FaceFeatureExtractor
from ailias.imageprocessing.face_grouping import group_face_images
from ailias.imageprocessing.faceparsing import load_verified_parser_model
from ailias.imageprocessing.images import ImageProcessor
from ailias.imageprocessing.matting import verified_model as load_verified_matting_model
from ailias.pipelines.photo_evidence_data import PhotoEvidenceBatch, PhotoEvidenceFrame
from ailias.storage import atomic_json


def create_photo_image_processor(store, device, gpu_index):
    """Create lazy preparation from verified local MODNet/BiSeNet assets.
    Preserve model/license provenance without installing or loading networks.
    """
    modnet_path, modnet_record = load_verified_matting_model(store)
    bisenet_path, bisenet_record = load_verified_parser_model(store)
    return ImageProcessor(
        modnet_model=modnet_path,
        bisenet_model=bisenet_path,
        device=device,
        gpu_index=gpu_index,
    ), {"modnet": modnet_record, "bisenet": bisenet_record}


def export_photo_image(store, image, target, artifacts):
    """Export one owned PNG and immediately register its immutable artifact reference.
    Never overwrite a source image or rely on inferred output filenames downstream.
    """
    image.save(target, format="PNG")
    reference = store.reference_artifact(target)
    artifacts[target.name] = reference
    return reference


def extract_photo_frame_features(
    store, capture, output, processor, extractor, artifacts
):
    """Prepare and extract one actual fitting grid, retaining original/crop provenance separately.
    Use the shared grouper's keep-all-usable decision without retaining live pixels in the result.
    """
    view_id = validate_identifier(capture["face_id"])
    original_path = store.artifact_path(capture["image"])
    if not store.verify(capture["image"]):
        raise ValueError("Retained sampled image is missing or changed: " + view_id)
    with ExitStack() as resources:
        image = resources.enter_context(processor.load_image(original_path))
        view_image = capture["image"]
        if capture.get("reviewed_crop") is not None:
            image = resources.enter_context(
                processor.crop_image(image, tuple(capture["reviewed_crop"]))
            )
            view_image = export_photo_image(
                store,
                image,
                output / (view_id + "-source.png"),
                artifacts,
            )
        prepared = processor.process_image(image)
        for value in {
            id(value): value
            for value in prepared.values()
            if isinstance(value, Image.Image)
        }.values():
            resources.callback(value.close)
        prepared.update(
            source=original_path,
            frame_index=capture.get("decode_index"),
            timestamp_seconds=capture.get("timestamp_seconds"),
        )
        extraction = extractor.extract_feature_result(prepared)
        grouped = group_face_images([extraction], review_directory=output / "review")
        category, members = next(iter(grouped.groups.items()))
        decision = members[0]
        regions = export_photo_image(
            store,
            prepared["regions"],
            output / (view_id + "-regions.png"),
            artifacts,
        )
        head = export_photo_image(
            store,
            prepared["image"],
            output / (view_id + "-head.png"),
            artifacts,
        )
        review_copy = None
        if decision.review_copy is not None:
            review_copy = store.reference_artifact(decision.review_copy)
            artifacts[Path(decision.review_copy).name] = review_copy
        return PhotoEvidenceFrame(
            view_id=view_id,
            capture=dict(capture),
            image=view_image,
            features=extraction.features,
            regions=regions,
            head_image=head,
            selected=decision.selected,
            view_group=category,
            reason=decision.exclusion_reason,
            error_type=extraction.error_type,
            execution={
                "preparation": prepared.get("execution", {}),
                "features": extraction.execution,
            },
            review_copy=review_copy,
            review_copy_error=decision.review_copy_error,
        )


def extract_photo_feature_evidence(
    store,
    samples,
    root,
    landmarker,
    *,
    device="cuda",
    gpu_index=0,
    quality_policy=None,
    processor=None,
    extractor=None,
) -> PhotoEvidenceBatch:
    """Build fresh typed photo evidence using the shared preparation/extraction APIs.
    Process one image at a time; own only models created here and exports under this job.
    """
    samples = list(samples)
    identifiers = [validate_identifier(sample["face_id"]) for sample in samples]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Photo evidence requires nonempty, unique view IDs")
    output = root / "photo-evidence"
    output.mkdir(exist_ok=False)
    manifest = output / "manifest.json"
    batch = PhotoEvidenceBatch(models={"mediapipe": {"model": landmarker}})
    owns_processor, owns_extractor = processor is None, extractor is None
    try:
        if processor is None:
            processor, models = create_photo_image_processor(store, device, gpu_index)
            batch.models.update(models)
        if extractor is None:
            extractor = FaceFeatureExtractor(
                mediapipe_model=store.artifact_path(landmarker),
                device=device,
                quality_policy=quality_policy,
            )
        for capture in samples:
            try:
                frame = extract_photo_frame_features(
                    store,
                    capture,
                    output,
                    processor,
                    extractor,
                    batch.artifacts,
                )
            except (OSError, ValueError, RuntimeError) as error:
                raise RuntimeError(
                    "Photo processing failed: " + capture["face_id"]
                ) from error
            batch.frames.append(frame)
            atomic_json(manifest, batch.to_dict())
        dependencies = [capture["image"] for capture in samples]
        dependencies.extend(iter_artifact_references(batch.models))
        if not all(store.verify(reference) for reference in dependencies):
            raise ValueError("Photo source or model changed during extraction")
        batch.status = "photo_evidence_ready"
    except Exception as error:
        batch.status = "failed"
        batch.error_type, batch.error_message = type(error).__name__, str(error)
        raise
    finally:
        for resource, owned in (
            (extractor, owns_extractor),
            (processor, owns_processor),
        ):
            if resource is not None and owned:
                try:
                    resource.close()
                except Exception as error:
                    batch.cleanup_errors.append(type(error).__name__)
        if batch.cleanup_errors:
            batch.status = "failed"
        primary = sys.exception()
        try:
            atomic_json(manifest, batch.to_dict())
        except OSError as error:
            if primary is None:
                raise
            primary.add_note(
                "Photo manifest persistence also failed: " + type(error).__name__
            )
    if batch.status != "photo_evidence_ready":
        raise RuntimeError(
            "Photo processing resource cleanup failed; inspect " + str(manifest)
        )
    return batch

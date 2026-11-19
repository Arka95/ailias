[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Image preparation and likeness metrics

## Photo-to-MetaHuman investigation

[Route and feature research](photo-to-metahuman-research.md) separates photo
observations from native control/UV translation. Vision-language models may
describe appearance, but descriptions are not native DNA or measured geometry.

## Model choices

| Task | Choice | Why / limit | Primary source |
|---|---|---|---|
| Boxes and five landmarks | YuNet | Small OpenCV model; profiles/occlusions can be missed | [OpenCV face tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html) |
| Candidate identity grouping | SFace normalized cosine | Identity signal, not facial dimensions or calibrated probability | Same tutorial |
| Visible-person extraction | PP-HumanSeg | Person foreground, not per-person instances or head-only matting | [Pinned model](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/human_segmentation_pphumanseg) |
| HEIC decoding | pillow-heif | Pillow integration with full primary-image decoding | [Plugin documentation](https://pillow-heif.readthedocs.io/en/latest/pillow-plugin.html) |
| Head isolation | Semantic face regions plus opacity | Removes neck/clothing while retaining soft hair edges; predictions can fail | [Implementation contracts](dev.readme.md) |

Model identities/checksums and license files are retained beside prepared
weights; processing does not download replacements.
OpenCV's reported LFW cosine threshold is not a universal alias/quality gate.
The configured complete-link threshold needs calibration on reviewed examples.

Generative restoration and aggressive denoising are not reconstruction evidence.
Preparation preserves original pixels and limits derived products to documented
orientation, cropping, masking and bounded colour transformations.

## What facial similarity can mean

| Signal | Use | Limitation |
|---|---|---|
| Identity cosine | Candidate grouping | Not anatomy, physical dimensions or 3D likeness |
| Projected landmark error | Pose-aware image comparison | Depends on camera, expression, correspondence and alignment |
| Corresponding 3D landmarks | Rigid/no-reflection sparse geometry comparison | Does not cover the whole face |
| Sampled Chamfer/Hausdorff/p95 | Bidirectional supplied-sample distance | Sensitive to crop/density; not exact mesh-surface evaluation |
| Scan-to-mesh evaluation | Research comparator | Requires appropriate reference geometry and alignment |

The geometry module uses NumPy SVD and forbids reflection.
Rigid alignment preserves physical size; similarity alignment can hide scale
error. No arbitrary 0-100 likeness percentage is defined.
Matched-camera silhouette/reprojection and animation remain independent checks.

Primary evaluation references:
[NoW](https://now.is.tue.mpg.de/),
[REALY](https://arxiv.org/abs/2203.09729),
[Open3D ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html),
[SciPy Hausdorff](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html).

## Curated portrait filtering

| Choice | Reason / limit |
|---|---|
| Individual reviewed SFace exemplars | Preserve pose variation without another model stack |
| Exact reviewed labels first | Do not regenerate rejected crops to infer their label |
| Positive/negative margins plus quality | Similarity alone does not establish valid content or sharpness |
| Abstention band | Unknown/conflicting examples go to review, not forced deletion |
| No automatic relabeling | Predictions do not become training labels |

Human-kept crops remain available even when they fail downstream reconstruction
quality. New rejection requires strong negative support, weak positive support
and margin. Normalized SFace magnitude is not a quality score; MagFace learns
different model-specific behavior.

References: [SFace](https://arxiv.org/abs/2205.12010),
[MagFace](https://arxiv.org/abs/2103.06627),
[QMagFace](https://arxiv.org/abs/2111.13475) and
[decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html).

## Runtime verification

Explicit CUDA selection requires completed inference and backend graph evidence,
not advertised hardware capability. It does not imply image I/O/grouping run on
the GPU. Do not silently retry on CPU or replace the shared runtime.
See [tests and integration boundary](dev.readme.md#validation).

[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Imageprocessing

Prepare images/video frames and extract one known person's facial measurements
and appearance. This module does not construct a head, fit a 3D model or operate
an editor. Original files remain unchanged; exports are explicit.

## One image through the two CNNs

`ImageProcessor.process_image()` runs the following preparation steps. A CNN is
a trained convolutional neural network; **MODNet and BiSeNet are the two
preparation CNNs**, not the later facial-landmark model.

| Step | Input and operation | Output passed forward |
|---|---|---|
| Load | Decode the original with EXIF orientation and colour-profile handling; keep supplied transparency | Native-size image and an unchanged RGB reference |
| **1. MODNet: remove background** | Composite existing transparency onto neutral grey; letterbox to the model's 512-square input; predict opacity and resize it back | RGBA person foreground with soft alpha. Neck and clothing can still remain; existing alpha is never restored |
| **2. BiSeNet ResNet18: isolate head** | Composite the MODNet result onto neutral grey; infer 19 semantic classes at 512 square; restore labels with nearest-neighbour resizing | Native-size label image plus RGBA head. Face/ears/hair and the parser's glasses class are retained; neck/clothing/background are removed |
| Optional colour normalization | Estimate bounded white balance from original scene neutrals outside the head; skip unsupported estimates | Corrected head RGB with unchanged alpha; original RGB remains available for measurements |

BiSeNet receives **the grey-composited MODNet result**, not a second independent
copy of the original. Neither CNN crops, aligns or frontalizes the returned
image. Both predictions are restored to the input pixel grid.

| Existing original | Example head output after MODNet and BiSeNet |
|---|---|
| ![Original portrait](images/original.png) | ![Head-only output from the existing full-image MODNet and BiSeNet experiment](images/modnet-bisenet.png) |

These are retained local demonstration images, **not a newly executed run**.
The cutout comes from the full-image MODNet -> grey composite -> BiSeNet
experiment. White balance and the later MediaPipe stage are not demonstrated
in these previews. The [illustrated guide](illustrated-pipeline.md) also labels
the separate older crop/detection examples; those are not steps of this API.

Preparation returns these values **in memory**, without saving sidecars:

| Field | Contents |
|---|---|
| `image` | Native-size RGBA head, optionally colour-normalized |
| `original_image` | Original RGB reference on the same oriented grid, before normalization |
| `regions` | Grayscale image of semantic class IDs, not an RGB preview or per-class probability maps |
| `colour` | White-balance outcome and gains, or a skipped/not-applied result |
| `status`, `quality` | `ready` means nonempty head foreground, not approval of facial quality |
| Source/execution fields | Source path, optional video frame/index time, and requested/backend execution details |

## From prepared appearance to FaceFeatures

`FaceFeatureExtractor` consumes the **existing head image, original RGB and
BiSeNet labels together**. It does not repeat either preparation CNN.

1. `extract_bisenet_features(original_rgb, regions)` measures natural-region
   appearance: pixel coverage, bounds, usable colour samples and review flags.
   This is pixel measurement using existing labels, **not another CNN**.
2. MediaPipe receives the prepared RGBA head composited onto neutral grey and
   predicts 478 landmarks, expression scores and a facial transform.
3. Landmarks map onto the unchanged source grid. `photo_features()` combines
   them with original RGB and the **already measured** BiSeNet region record
   to calculate projected facial ratios and landmark-guided iris colour.
4. The extractor assembles `FaceFeatures`, adding orientation and quality
   annotations. Regional colours do not feed into or refine MediaPipe.

No separate appearance pass needs to run before calling the extractor: it owns
steps 1-4. Missing/ambiguous landmark detection can leave partial regional
appearance without inventing facial geometry.

## Simple Python API

**Single-person scope:** every input belongs to the same known person. Identity
embeddings, person clustering and subject matching are intentionally unnecessary
in this API. Grouping means organizing views, not identifying people.
Every usable extraction is selected, with no per-view quota. Quality warnings
annotate images; they do not reject useful side/upper/lower views.

```python
from pathlib import Path
from ailias.imageprocessing.face_feature_extraction import FaceFeatureExtractor, FaceFeatures
from ailias.imageprocessing.face_grouping import group_face_images
from ailias.imageprocessing.images import ImageProcessor

models = Path(r"data\arka\imageprocessing\models")
with ImageProcessor(
    modnet_model=models / "modnet-photographic-v1" / "modnet-512.onnx",
    bisenet_model=models / "face-parsing-resnet18-v1" / "resnet18.onnx",
) as processor:
    prepared = processor.process_image(Path(r"C:\photos\face.jpg"))

with FaceFeatureExtractor(
    mediapipe_model=Path(r"data\arka\facefitting\models\mp-face-landmarker-v1\face_landmarker.task"),
) as extractor:
    extractions = list(extractor.extract_features_batch([prepared]))

face_features: FaceFeatures | None = extractions[0].features
if face_features is not None and face_features.bisenet is not None:
    region_data = face_features.bisenet.to_dict()  # Nested dictionary.
    region_json = str(face_features.bisenet)  # JSON string; no file writes.
if face_features is not None and face_features.quality is not None:
    print(face_features.orientation, face_features.quality.reasons)
grouped = group_face_images(
    extractions, review_directory=Path(r"data\arka\imageprocessing\review"),
)
print(len(grouped.selected), len(grouped.excluded))
prepared["image"].save("head.png")  # Separate, explicit head export.
```

The default is `device="cuda", gpu_index=0`; use `device="cpu"` only for explicit
CPU-only processing. On Windows in CUDA mode, **only MediaPipe uses CPU**, because
its Python GPU delegate is unsupported there. Elsewhere CUDA mode requests the
MediaPipe GPU delegate. Unsupported requested GPU execution fails; no CPU retry.

## What FaceFeatures contains

`extract_features(prepared)` returns a `FaceFeatures` instance containing facial
data only. Its fields describe one view, not an identity model or a constructed head.

| Field | Contents and limits |
|---|---|
| `source_size` | `(width, height)` of the landmark pixel grid, including any explicit caller crop |
| `landmarks` | `normalized_predicted`: 478 x 3 raw MediaPipe predictions; `source_xy`: 478 x 2 pixel coordinates; `on_head`: 478 in-image/positive-alpha support flags. Predicted z is not metric depth |
| `bisenet` | Typed `FaceFeaturesBiseNet`: natural-region appearance records described below, not dense pixel masks |
| `features` | Dictionary containing `geometry_2d` pixel lengths/eye-span ratios, `regions` appearance summaries, separate `iris` evidence and heuristic `facial_hair_appearance` when skin colour is measurable, plus limitation flags |
| `blendshapes` | Per-view expression scores, not shared identity or a neutral mesh |
| `facial_transform` | Raw model transform, when available |
| `orientation` | Approximate pitch/yaw/roll or a missing/invalid-pose diagnostic; not calibrated anatomical direction |
| `quality` | Typed `FaceQuality`: detail, exposure, support, clipping, expression and warning reasons |
| `features_colour_source` | `"original_rgb"`: colours retain the photographed lighting, not the normalized cutout colours |

`FaceFeatures` contains **no image pixels, alpha image, dense parser masks,
source path or selection decision**. For grouping, use `extract_feature_result(prepared)`
for one fail-fast extraction, or batch extraction for per-image failure isolation:
its `FaceExtractionResult` items keep facial data in `.features`, original/head
Pillow references in `.image`/`.head_image`, and source/frame/backend/error
metadata alongside it. Wrapper JSON omits the image pixels.
`FaceGroupingResult` owns `.groups`, `.selected` and `.excluded`; grouping never
recalculates orientation or quality.
`mediapipe_model` and `quality_policy` belong to the extractor; view-bin policy
belongs to grouping.

`face_features.quality` is a `FaceQuality` instance with `native_short_side`,
`laplacian_variance`, `dark_fraction`, `bright_fraction`, `supported_landmarks`,
`clipped_landmarks`, `max_expression_score` and warning `reasons`.
Counts refer to the 468 base-face landmarks, excluding iris points. It supports
`to_dict()` and `str(...)` for dictionary/strict JSON output. Quality is `None`
when facial extraction cannot produce an assessment; crop metrics remain `None`
for entirely outside or degenerate face bounds. Missing values are not invented.
Grouping's `.quality` and `.orientation` properties expose the same extracted
records, not new measurements or independent copies.

`face_features.bisenet` is a `FaceFeaturesBiseNet` instance with named region
objects: `skin`, `left_eyebrow`, `right_eyebrow`, `left_eye`, `right_eye`,
`left_ear`, `right_ear`, `nose`, `mouth_interior`, `upper_lip`, `lower_lip`
and `scalp_hair`. Combined `eyebrows`/`lips` summaries remain available.
Each retains parser IDs, region/sample counts, pixel bounds, median linear-RGB
colour when measurable, and review/provenance fields. Missing parts request
review instead of inventing colours. Accessories, neck and clothing are excluded.
Eye-region colour is not a separate iris or pupil measurement.

For example, one **illustrative, not measured-from-the-previews** skin record is:

```json
{
  "status": "observed",
  "parser_classes": [1],
  "region_pixels": 18000,
  "sample_pixels": 15600,
  "bounds_xy": [[40, 35], [180, 210]],
  "median_linear_rgb": [0.20, 0.13, 0.10],
  "photographed_lighting_retained": true
}
```

`region_pixels` counts the region before colour-sample filtering;
`sample_pixels` counts usable, non-clipped colour samples. The RGB values are
independent normalized linear-light channel intensities, **not percentages
that add up to 100**. Missing/insufficient regions retain review evidence
instead of a fabricated colour. Beard, moustache and eyelashes have no separate
class in the current parser. `features["facial_hair_appearance"]` separately
estimates beard/moustache darkness inside landmark regions relative to observed
skin colour; shadows and pigmentation remain ambiguous. This is not beard
segmentation, an eyelash measurement or intrinsic reflectance.

These are derived pixel statistics, not retained raw CNN score maps. They reuse
existing labels and do not feed into or refine MediaPipe. The enclosing
`FaceFeatures` also supports `to_dict()` and `str(...)` for dictionary/JSON output.
`FaceFeatures.from_dict(data)` restores typed regions/quality from saved JSON without
inference or image loading. `source_size` belongs to the landmark coordinate frame,
not source identity: it is the width/height of the prepared, EXIF-oriented pixel
grid, including any explicit caller crop. It contains no RGB/alpha pixels.

**Next step:** selected `FaceFeatures` supply dimensions, 2D landmarks and support
to the separate [head-reconstruction fitter](../head_reconstruction/README.md#reconstruct-from-retained-facefeatures),
which owns shared-identity fitting and neutral-head construction.

Old bare feature records without dimensions are not independently usable for fitting;
recover the matching grid dimensions from their extraction wrapper, not by guessing.

## View grouping and runtime

Colour normalization is enabled by default for the returned cutout, not feature
samples; pass `normalize_colour=False` to skip it. Original RGB supplies the
uncorrected colour reference and feature-stage image-quality assessment; MediaPipe does not require it.
This is bounded scene-neutral
white balance, not calibrated skin colour or cross-photo exposure matching.
No/ambiguous detections or invalid dense points are extraction failures. Grouping
also excludes missing facial data, degenerate bounds or faces entirely outside
the image. Low detail/exposure, partial clipping, weak alpha support and expressions
are warnings only. Region/iris features can independently require review.

Orientation calculation never discards images or raises a per-image pose error:
unavailable or invalid transforms produce an orientation diagnostic
`reason="missing_or_invalid_pose"`; grouping places these images in `unknown`.
Otherwise bins include `front`, `left_oblique`,
`right_oblique`, `left_profile`, `right_profile`, `up`, `down` and combined views
such as `left_profile_up`. Angles are approximate model-coordinate conventions,
not certified anatomical side or camera elevation; roll is retained, not penalized.
Positive uniform transform scale is removed before rotation validation.

Excluded candidates remain in their view group with `.selected=False` and an
`.exclusion_reason`. Only their **copies** go into the explicit `review_directory`:
source-photo bytes are copied unchanged; decoded video frames and in-memory inputs
are saved as PNG. Originals are never moved/deleted, existing review files are
never overwritten, and unavailable copies expose `.review_copy_error` rather than
silently disappearing. Selected images create no review files.

For a lazy feature batch, pass `processor.process_images(paths)` into
`extractor.extract_features_batch(...)` while both contexts are open. Models are
reused; preparation and extraction failures become per-item `FaceExtractionResult`
records. Pass the resulting iterator to `group_face_images(..., review_directory=...)`
before closing the contexts. Grouping retains candidate pixels in memory, so bound
large video batches with `max_frames` or process smaller collections.
`processor.extract_video_frames(path)` yields `Frame`
records with RGB, source, decode index and timestamp; pass these records directly
to `process_image()` or `process_images()`. Sampling defaults to one frame per
second; `interval_seconds` and `max_frames` control it. Close an unfinished frame
iterator to release its decoder. `load_image()` preserves orientation, colour
profiles and existing transparency; `crop_image()` accepts explicit pixel bounds.

Models load lazily on the selected backend from supplied local files, with no
downloads, external processes or global thread-setting changes. Use prepared model files
compatible with the existing MODNet static-512 and BiSeNet ResNet18 adapters.
Preparation needs the existing `imageprocessing` extra; feature extraction also
needs `dense-landmarks`. CUDA inference requires the existing CUDA-enabled OpenCV environment;
do not replace it with a CPU-only wheel. Frame extraction also needs PyAV
from the existing `video` extra.
Use each instance serially, not concurrent calls sharing mutable model state.
No alpha sidecars, JSON or previews are retained automatically; grouping writes
only the explicitly requested excluded-image review copies.

## Existing CLI inputs and outputs

The commands below still use the older production path during the staged
migration; they do **not yet** compose the Python API described above.
Their identity grouping, alignment and stricter selection are not prerequisites
of the new single-person API.

| Input | Output | Consumer |
|---|---|---|
| File/folder: JPEG, PNG, WebP, TIFF, HEIC/HEIF/HIF | Extraction manifest, source hashes, face boxes, embeddings and group contact sheets | Curation, selection and cleansing |
| Native portrait + face semantics + opacity | Head-only RGBA, background-suppressed RGB, alpha and reversible aligned analysis image | Facefitting and generation |
| Explicit keep/remove review | Curation profile; uncertain new crops go to a review directory | Future scans and downstream input validation |
| Selected group | Selection manifest referencing existing portraits/originals | Core `intake`; does not attest capture conditions |
| Prepared photo inputs + existing dense observations | Per-photo shape ratios, iris/skin/brow/lip/hair colours, parser labels and provenance | Native diagnostic comparison; not intrinsic albedo or calibrated 3D |

Artifacts live below `data\ALIAS\imageprocessing`: reusable `assets`, immutable
`runs`, versioned `cleansing` jobs and `selections`. Native RGB/alpha preserve
the portrait grid; aligned images are analysis-only, not frontalized reconstruction inputs.

## Usage

Use the shared environment from the [root setup](../../README.md#quick-setup).
Model preparation is explicit; `scan` never downloads weights.

```powershell
python -m ailias imageprocessing models subject
python -m ailias imageprocessing scan subject C:\photos --device cuda
python -m ailias imageprocessing curate subject RUN_ID --removed-are-rejected
python -m ailias imageprocessing select subject RUN_ID face-001
python -m ailias imageprocessing cleanse subject RUN_ID --group face-001 --device cuda
```

Inspect contact sheets and crops before `curate` or `select`. `curate` records
explicit review, not automatic predictions: keep valid portrait PNGs, remove
invalid ones, then confirm removals with its flag. Uncertain files can be moved
unchanged from `assets\review` to `assets\portraits` before curation.

| Option | Default | Meaning |
|---|---|---|
| `scan --device`, `cleanse --device` | `cpu` | `cpu` or `cuda`; CUDA failure does not retry on CPU |
| `scan --recursive` | Off | Traverse subdirectories |
| `scan --limit N` | All supported files | Bound the sorted input list |
| `--no-body` | Off | Skip optional person segmentation; supported by `models` and `scan` |
| `--no-cleanse` | Off | Skip head-matting/parser preparation or cleansing, respectively |
| `cleanse --group GROUP` | Per stored group | Share exposure references across a supported seed-subject hypothesis |
| `scan / curate --processing-config FILE` | `config\imageprocessing.json` | Select extraction/curation thresholds |
| `scan / cleanse --cleansing-config FILE` | `config\cleansing.json` | Select head-alpha and colour policy |

CPU inference uses the optional `imageprocessing` extra. Do not install its
headless OpenCV wheel over an existing CUDA-enabled OpenCV installation.
Matting setup requires the optional `matting-tools` extra; see [runtime setup](dev.readme.md#runtime).

## Configuration

### Feature-extraction quality: `config\face_quality.json`

`FaceFeatureExtractor` loads this file unless given an explicit
`quality_policy=FaceQualityPolicy(...)`. Use `load_face_quality_policy(path)`
from `ailias.imageprocessing.settings` for another file. These thresholds
annotate measurements; they do not exclude images.

| Key | Default | Meaning |
|---|---|---|
| `min_face_size` | `256` | Native face-box short side in pixels |
| `min_laplacian_variance` | `40.0` | Focus warning threshold after bounded downsampling |
| `max_dark_fraction` / `max_bright_fraction` | `0.5` / `0.5` | Allowed fractions of pixels below 16 / at least 240 |
| `min_landmark_alpha` | `240` | Minimum 0-255 opacity for quality-support counting |
| `min_supported_landmarks` | `150` | Required supported points among the 468 base-face predictions |
| `max_neutral_expression` | `0.5` | Expression/jaw score warning threshold |

### Single-person grouping: `config\face_grouping.json`

`group_face_images()` loads this file unless given an explicit `FaceGroupingPolicy`.
Use `load_grouping_policy(path)` for another file. View boundaries classify
already extracted angles, not acceptability. Quality settings moved from this
file/policy into the extractor's separate quality configuration above.

| Key | Default | Meaning |
|---|---|---|
| `frontal_yaw_degrees` | `20.0` | Maximum absolute frontal-bin yaw |
| `profile_yaw_degrees` | `55.0` | Minimum absolute profile-bin yaw; intervening views are oblique |
| `vertical_pitch_degrees` | `20.0` | Absolute pitch above this adds up/down classification |

### Existing CLI extraction and curation: `config\imageprocessing.json`

| Key | Default | Meaning |
|---|---|---|
| `detection_score` | `0.85` | Minimum YuNet detector confidence |
| `detection_max_side` | `1600` | Longest analysis side in pixels; native coordinates are restored |
| `cpu_threads` | `2` | OpenCV worker-thread budget |
| `min_face_size` | `256` | Minimum native face-box side in pixels |
| `min_laplacian_variance` | `40` | Minimum face-region focus score after bounded downsampling |
| `max_dark_fraction` | `0.5` | Maximum dark-pixel fraction in the face quality test |
| `max_bright_fraction` | `0.5` | Maximum bright-pixel fraction in the face quality test |
| `identity_cosine_threshold` | `0.5` | Minimum pairwise SFace cosine for complete-link grouping |
| `identity_ambiguity_margin` | `0.05` | Near-tie margin used to flag ambiguous identity assignments |
| `portrait_side_margin` | `0.6` | Extra crop margin per side, relative to face-box width |
| `portrait_top_margin` | `0.75` | Extra upper crop margin relative to face-box height |
| `portrait_bottom_margin` | `1.0` | Extra lower crop margin relative to face-box height |
| `curation_accept_cosine` | `0.5` | Minimum similarity to kept examples for automatic acceptance |
| `curation_reject_cosine` | `0.6` | Minimum similarity to rejected examples for automatic rejection |
| `curation_reject_positive_ceiling` | `0.35` | Rejection also requires similarity to kept examples below this value |
| `curation_score_margin` | `0.1` | Required winning margin between positive and negative exemplar scores |

Exact reviewed labels take priority. New acceptance also requires quality gates;
new rejection needs all negative-score, positive-ceiling and margin tests.
Cosines are not calibrated probabilities.

Inactive `min_landmark_alpha`, `min_supported_landmarks`, `max_neutral_expression`,
`max_pitch_roll_degrees` and `max_yaw_degrees` defaults were removed from this
configuration. `ProcessingPolicy` still accepts the historical fields because
saved CLI manifests are deserialized by cleansing, generation and reconstruction.
They do not control new extraction/grouping; active quality thresholds belong
in `face_quality.json`. Legacy modules remain because their CLI/downstream
callers are live; this change does not retire or migrate those commands.

## Portrait background removal and colour consistency

### Cleansing: `config\cleansing.json`

| Key | Default | Meaning |
|---|---|---|
| `neutral_saturation_max` | `0.2` | Maximum saturation for scene-neutral white-balance samples |
| `neutral_min_fraction` | `0.005` | Minimum fraction of usable scene-neutral pixels |
| `white_balance_gain_limit` | `1.18` | Maximum permitted per-channel gain or inverse gain |
| `exposure_ev_limit` | `0.35` | Maximum absolute exposure adjustment in stops |
| `alpha_background_cutoff` | `0.02` | Snap alpha at/below this threshold to fully transparent |
| `alpha_foreground_cutoff` | `0.98` | Snap alpha at/above this threshold to fully opaque |
| `native_mask_threshold` | `0.5` | Binary alpha threshold forwarded to native reconstruction |
| `min_face_coverage` | `0.9` | Minimum retained coverage of the face core |

BiSeNet selects face/ears/hair; MODNet supplies soft opacity. Neck, clothing and
background are excluded. Colour correction uses bounded scene-neutral estimates
and subject-relative exposure, not calibrated skin colour, shadow removal or albedo.
Insufficient colour evidence is skipped; unreliable masks require review.

## Review and limitations

| Boundary | Behavior |
|---|---|
| Identity | Complete-link grouping may split poses or confuse lookalikes; inspect candidate groups |
| Curation | Explicit kept files are reused unchanged; changed/missing references fail closed |
| Coordinates | Eye-axis alignment changes roll/scale/position only; no synthesized yaw/pitch or hidden anatomy |
| Cleansing | Hair, glasses, accessories and touching subjects can be misclassified |
| Selection | Excludes unsupported identities, quality failures and nonaccepted curation decisions |
| Fresh runs | Inference runs again; content-addressed PNG reuse is storage deduplication, not inference replay |

For comparison:
`python -m ailias imageprocessing compare subject RUN_ID FIRST_FACE_ID SECOND_FACE_ID`.
This separates identity cosine from five-point projected geometry.

## Cleanup

```powershell
python -m ailias cleanup --alias subject
python -m ailias imageprocessing cleanup subject
python -m ailias imageprocessing cleanup subject --prune-superseded --keep-run RUN_ID
```

Job finalizers remove owned unreferenced scratch. Explicit legacy cleanup
consolidates generated assets and verifies originals before removing redundant
source copies. Referenced assets, curation examples and models are protected.
See [global cleanup safeguards](../dev.readme.md#resource-cleanup) and
[research](research.md).

## Reusable feature evidence

`features` explicitly reuses existing 478-point observations; it does not rerun
dense inference, fit geometry or average appearance across photos.
`inspect-image` performs fresh CPU landmark inference on one explicit image,
including an existing native render. Both use the installed face parser and
never download models or start Blender/Unreal.
Fresh dense inspection requires the `dense-landmarks` optional dependency
(also already provided by `facefitting`); reusing observations does not.

```powershell
python -m ailias imageprocessing features subject --inputs C:\jobs\sources.json --observations C:\jobs\observations.json --cohort frontal-reference --face-id SOURCE_FACE_ID --device cpu
python -m ailias imageprocessing inspect-image subject C:\renders\neutral.png --landmarker C:\models\face_landmarker.task --cohort native-draft --role render --device cpu
```

| Parameter / output | Meaning |
|---|---|
| `--inputs` | Existing version-1 prepared `sources.json`; current curation hash and selected source/crop/alignment dependencies must match |
| `--observations` | Existing `mediapipe-478-reversible-source-v1` JSON; source IDs and coordinate transforms are checked |
| `--face-id` | Repeatable subset for `features`; omitted means all prepared faces, not a new identity selection |
| `--cohort` | Required appearance-reference label; descriptive, not automatic capture/date validation |
| `--landmarker` | Existing MediaPipe task file for `inspect-image`, hash-pinned in the output |
| `--role` | Required `reference` or `render` for a single-image inspection; prevents treating another photo as native output |
| `--device` | Parser device, default `cpu`; explicit `cuda` never falls back. Dense inspection remains CPU |
| `geometry_2d` | Pixel lengths and outer-eye-span-normalized ratios; pose/expression-dependent, not mm |
| `regions` / `iris` | Linear-RGB medians, usable sample counts, masks/bounds and observed/review status; lighting retained |
| `source_xy` | Full source-photo coordinates; region masks use the recorded portrait origin |
| `unresolved` | Beard, moustache and lashes: no distinct class in the current parser, not silently mapped to scalp hair |

Model-specific sufficiency rules are internal constants, not deployment settings:
geometry needs 12px outer-eye span; regions need 32 unclipped pixels; iris uses a
0.45-0.85 annulus, at least 12 usable pixels before luminance trimming and 8 after.
Channels at/below 2 or at/above 250 are excluded from colour samples.
Head-mask support is not independent occlusion detection.

Jobs under `imageprocessing\features\REVISION` retain a manifest and region-label
PNGs. `features_extracted_awaiting_review` does not mean every region is observed.
Missing/multiple faces require review; changed inputs fail rather than producing
a native edit. Compare with [facefitting](../facefitting/README.md#photo-native-feature-comparison).

[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Imageprocessing

Extract reviewable face groups, portraits and head-only foreground from photos.
Original files remain unchanged. The new API groups one person's views;
the existing CLI retains its heuristic identity grouping and quality gates.
Reusable photo-feature extraction belongs here; native asset/control translation
belongs downstream. Regional geometry/colour/iris evidence does not include
separate beard/lash segmentation or automatic native likeness fitting.
See the [research and limitations](photo-to-metahuman-research.md).
For model-by-model inputs, outputs and actual local portrait examples, see the
[illustrated image-to-features pipeline](illustrated-pipeline.md).

## Simple Python API

`images.py` prepares pixels with **MODNet -> BiSeNet -> optional white balance**.
`face_feature_extraction.py` uses **MediaPipe -> facial features**, reusing the
prepared head and labels: no second matting, parsing or head-isolation pass.
`face_grouping.py` determines **orientation -> view groups -> selection**.
These stages use no YuNet, SFace, automatic cropping or alignment. Existing commands
below are unchanged and still use their original pipeline.

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

| Stage | Result fields |
|---|---|
| Preparation | `image`: native-size RGBA head; `original_image`: untouched RGB on the same grid; `regions`: semantic labels; `colour`: white-balance outcome |
| `FaceFeatures` | `landmarks`, `blendshapes`, raw `facial_transform`, `features`: facial ratios/iris evidence; `bisenet`: natural-region appearance; colour-source semantics |
| Batch `FaceExtractionResult` | Facial data in `.features`; source/frame metadata, original/prepared pixels, execution provenance and optional `.error_type`/`.reason` |
| `FaceGroupingResult` | `.groups`: all candidates by view, including excluded members; `.selected`/`.excluded`: decision lists with orientation, quality warnings and review-copy outcome |

Preparation `ready` means a nonempty head, **not face-quality approval**.
The extractor records measurements and raw model predictions; grouping owns
orientation, image-quality assessment and keep/exclude decisions.
`extract_features(prepared)` returns a **`FaceFeatures` instance** containing
facial data only. It can contain partial BiSeNet evidence without landmarks.
For grouping, use batch extraction: **its items are now `FaceExtractionResult`**,
not `FaceFeatures`. Source/frame/backend/error fields moved into this wrapper.
`mediapipe_model` belongs to the extractor; quality/view policy belongs to grouping.

`face_features.bisenet` is a `FaceFeaturesBiseNet` instance with named region
objects: `skin`, `left_eyebrow`, `right_eyebrow`, `left_eye`, `right_eye`,
`left_ear`, `right_ear`, `nose`, `mouth_interior`, `upper_lip`, `lower_lip`
and `scalp_hair`. Combined `eyebrows`/`lips` summaries remain available.
Each retains parser IDs, region/sample counts, pixel bounds, median linear-RGB
colour when measurable, and review/provenance fields. Missing parts request
review instead of inventing colours. Accessories, neck and clothing are excluded.
Eye-region colour is not a separate iris or pupil measurement.

These are derived pixel statistics, not retained raw CNN score maps. They reuse
existing labels and do not feed into or refine MediaPipe. The enclosing
`FaceFeatures` also supports `to_dict()` and `str(...)` for dictionary/JSON output.

Colour normalization is enabled by default for the returned cutout, not feature
samples; pass `normalize_colour=False` to skip it. Original RGB supplies the
uncorrected colour reference and later image-quality assessment; MediaPipe does not require it.
This is bounded scene-neutral
white balance, not calibrated skin colour or cross-photo exposure matching.
No/ambiguous detections or invalid dense points are extraction failures. Grouping
also excludes missing facial data, degenerate bounds or faces entirely outside
the image. Low detail/exposure, partial clipping, weak alpha support and expressions
are warnings only. Region/iris features can independently require review.

Orientation never discards images or raises a per-image pose error: unavailable
or invalid transforms produce `category="unknown"` with
`reason="missing_or_invalid_pose"`. Otherwise bins include `front`, `left_oblique`,
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

## Inputs and outputs

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

### New single-person grouping: `config\face_grouping.json`

`group_face_images()` loads this file unless given an explicit `FaceGroupingPolicy`.
Use `load_grouping_policy(path)` for another file. These warning thresholds do
not exclude usable extractions; view boundaries label orientation, not acceptability.

| Key | Default | Meaning |
|---|---|---|
| `min_face_size` | `256` | Native face-box short side in pixels |
| `min_laplacian_variance` | `40.0` | Focus warning threshold after bounded downsampling |
| `max_dark_fraction` / `max_bright_fraction` | `0.5` / `0.5` | Allowed fractions of pixels below 16 / at least 240 |
| `min_landmark_alpha` | `240` | Minimum 0-255 opacity for quality-support counting |
| `min_supported_landmarks` | `150` | Required supported points among the 468 base-face predictions |
| `max_neutral_expression` | `0.5` | Expression/jaw score warning threshold |
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
| `min_landmark_alpha` | `240` | Minimum 0-255 head opacity for a dense landmark to count as supported |
| `min_supported_landmarks` | `150` | Retained compatibility key; new API reads its separate grouping policy |
| `max_neutral_expression` | `0.5` | Retained compatibility key; new API reads its separate grouping policy |
| `max_pitch_roll_degrees` | `25.0` | Retained compatibility key; does not limit new view grouping |
| `max_yaw_degrees` | `70.0` | Retained compatibility key; does not limit new view grouping |
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

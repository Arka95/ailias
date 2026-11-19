[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Imageprocessing

Extract reviewable face groups, portraits and head-only foreground from photos.
Original files remain unchanged; identity grouping and quality gates are heuristics.
Reusable photo-feature extraction belongs here; native asset/control translation
belongs downstream. Regional geometry/colour/iris evidence does not include
separate beard/lash segmentation or automatic native likeness fitting.
See the [research and limitations](photo-to-metahuman-research.md).

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

### Extraction and curation: `config\imageprocessing.json`

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

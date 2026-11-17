[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Imageprocessing

Extract reviewable face groups, portraits and head-only foreground from photos.
Original files remain unchanged; identity grouping and quality gates are heuristics.

## Inputs and outputs

| Input | Output | Consumer |
|---|---|---|
| File/folder: JPEG, PNG, WebP, TIFF, HEIC/HEIF/HIF | Extraction manifest, source hashes, face boxes, embeddings and group contact sheets | Curation, selection and cleansing |
| Native portrait + face semantics + opacity | Head-only RGBA, background-suppressed RGB, alpha and reversible aligned analysis image | Facefitting and generation |
| Explicit keep/remove review | Curation profile; uncertain new crops go to a review directory | Future scans and downstream input validation |
| Selected group | Selection manifest referencing existing portraits/originals | Core `intake`; does not attest capture conditions |

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

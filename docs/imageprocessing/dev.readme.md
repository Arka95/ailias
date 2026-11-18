[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Imageprocessing internals

## Runtime

`scan` accepts a backend implementing detection, segmentation and provenance.
CLI mutations hold `Store.lock()`. CPU/CUDA OpenCV implementations share
preprocessing; model imports/handles are task-scoped and released on exit.

| Runtime | Contract |
|---|---|
| YuNet | Bounded analysis size; restore separate x/y scaling to native coordinates |
| SFace | Five-point alignment for embeddings only; exported portraits retain projection |
| PP-HumanSeg | Visible-person foreground; ambiguous multi-face components are not individual bodies |
| BiSeNet + MODNet | Semantic head selection plus soft opacity, restored to native portrait grid |
| CUDA backend | Explicit device 0, completed forward and backend graph evidence; no CPU retry |

No-face detections retry distinct native-bounded analysis sizes of 960, 640 and
320 after the initial size. Success/error stops retries; confidence is not lowered.
The optional `matting-tools` extra prepares a pinned static ONNX graph and checks
equivalence. Inference remains OpenCV, not an ONNX Runtime service.

## Data contracts

| Boundary | Invariant |
|---|---|
| Original -> manifest | External path, SHA256 and EXIF-oriented coordinates; originals are not copied |
| Manifest -> PNG assets | Content-addressed reusable derivatives; fresh inference is still required per scan |
| Grouping -> selection | Deterministic complete-link cosine, co-occurrence exclusions and ambiguity checks |
| Curation -> scan | Explicit reviewed exemplars, matching recognizer revision and unchanged kept files |
| Cleansing -> native tools | Unwarped RGB/alpha; analysis alignment has recorded inverse transforms |
| Completed cleansing -> pointer | Advance only after successful current-curation output; failures preserve prior pointer |

Complete-link grouping is quadratic in pairwise storage and worst-case cubic
in merge search; this is bounded personal-photo processing, not a large-scale
identity service.

## Curation and colour

Exact reviewed decisions precede similarity heuristics. New acceptance requires
positive support, margin and quality; new rejection requires strong negative
support, weak positive support and margin. Conflicts/unknowns abstain.
No automatic prediction becomes a training label.

Semantic head context excludes neck/clothing. Hair fringe is intersected with
opacity and never restores explicitly excluded semantic regions. Lower-context
leakage or poor face coverage produces review status.

White balance samples low-saturation nonface scene pixels and abstains when
ambiguous. Exposure uses a robust frontal-subject reference with bounded stops.
Neither estimates intrinsic albedo. Aligned 512-square heads correct roll,
translation and uniform scale, not yaw/pitch or expression.

## Configuration

The [module configuration tables](README.md#configuration) define processing
and cleansing keys/defaults. `ProcessingPolicy` and `CleansingPolicy` validate
numeric bounds; there are no environment-variable threshold overrides.
Model download identities are pinned in the model adapters, separate from policy.

## Feature extraction and native diagnostics

| Component | Responsibility / boundary |
|---|---|
| `dense.py` | Task-scoped CPU MediaPipe predictions and reversible source coordinates; no GNM dependency |
| `appearance_features.py` | Pure per-photo geometry ratios, semantic-region colour and pupil-excluding iris sampling |
| `feature_evidence.py` | Validate input hashes/curation/transforms; retain versioned features and parser masks |
| `facefitting\appearance\feature_comparison.py` | Downstream photo/render diagnostics and optional native eye-profile proposal; never edits a Character |

`features` requires explicit observation reuse; `inspect-image` performs fresh
dense inference on one image. Both parse actual native-resolution RGB, not the
aligned analysis colours, and retain source coordinates. Each output preserves
appearance cohorts without cross-photo averaging. Source changes fail closed;
no/multiple detected faces and insufficient individual features abstain.

Focused units cover colour clipping/iris pupil rejection, coordinate restoration,
stale curation/source changes, unknown/duplicate face IDs, mid-run invalidation,
single-render inspection and native proposal dependencies. They use synthetic
images/fake models; no model is loaded during units.

Real CPU checks on November 17, 2026 extracted the frontal source as
`4c9ac011a6fb45f496d1c18f0fc31573` and inspected the **existing rejected** native
frame as `8ef12704e8574dc8b6df97d015957517`, under `imageprocessing\features`.
No new render, geometry fit or native edit was made by these checks.
The combined evidence, compatibility, fitting, native-policy and review unit
selection passed **56 tests**; no UE integration or model benchmark was run.

## Research decisions in plain language

| Decision | Reason / limit |
|---|---|
| Separate recognition, quality and anatomy | A high identity cosine can still describe a blurred or unusable crop |
| Complete-link grouping plus separate subject association | Reduce identity mixing without forcing every pose into one cluster |
| Keep native and aligned images separate | Centering helps analysis; warped pixels would corrupt reconstruction cameras |
| Semantic head mask plus soft matting | Person segmentation intentionally retains clothing |
| Bound colour changes and abstain | Scene neutrality/relative skin observations are not calibrated skin reflectance |
| Store derived images once | Reduce storage without replaying inference or changing original pixels |

Detailed model sources, experiments and validation records: [research](research.md).

## Validation

```powershell
python -m unittest discover -s src\ailias\imageprocessing\tests -t src -v
```

Units cover decoding/EXIF, corrupt/duplicate input, grouping ambiguity,
co-occurrence, curation, unchanged pixels, source integrity, reversible alignment,
colour abstention, mask forwarding and cleanup. Fake backends replace models;
no user photos, network or GPU probes run.

Real-data validation has separately exercised CUDA extraction, head-only
cleansing, full preview pagination and downstream RGB hardlinks/native masks.
Those records do not certify every alpha edge, identity assignment or skin tone.
Keep real model execution explicit and separate from unit discovery.

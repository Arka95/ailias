[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Imageprocessing internals

## Simple API

`images.py` owns one `ImageProcessor` and the plain `Frame` data record.
MODNet predicts opacity; BiSeNet labels a grey composite of that result and
restricts its alpha; MediaPipe detects directly on the head cutout. The new API
does not use the legacy detector/grouping backend or depend on native-engine
modules. Each function has a short descriptive docstring naming its model where
applicable.

ONNX networks and the dense detector load lazily, are reused serially across
images/frames, and are released by `close()` or context exit. Source geometry
stays unchanged. White balance changes only the returned cutout; feature colours
are sampled from original RGB using restored semantic labels and dense points.
This keeps the existing feature interpretation without retaining manifests.
`read_oriented_image(..., preserve_alpha=True)` supports the new loader;
its default remains RGB for all existing callers.

`ImageProcessor` defaults to `device="cuda"` with explicit `gpu_index=0`.
MODNet/BiSeNet check CUDA availability and verify graph execution, failing if
OpenCV silently ran entirely on CPU. `device="cpu"` is the explicit CPU-only
mode. In CUDA mode, MediaPipe uses CPU on Windows and requests its GPU delegate
elsewhere; the selection reason is in `detect_face_landmarks()`'s docstring.
This is the owner's explicit Windows exception, not a failed-GPU retry.
See [MediaPipe's delegate support](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/base_options.py).
Existing callers of `DenseDetector(model)` keep their CPU default; the new API
always supplies the selected delegate.

Results and batch errors stay in memory. Individual region/iris review flags
remain separate from overall image-quality status. Predicted pose angles flag
absolute pitch/roll above 25 degrees or yaw above 70; expression scores above
0.5 flag non-neutral expression. These are review heuristics, not calibrated
depth, pose, skin colour or independent per-point visibility.
`Frame` sampling uses presentation timestamps with bounded lazy consumption,
never average FPS; missing/invalid timestamps and rotated inputs fail explicitly.
No weights, codecs, GPU probes, real photos or native editors run in its mocked unit tests.

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
| `dense.py` | Explicit MediaPipe CPU/GPU delegate; legacy CPU default and reversible source coordinates; no GNM dependency |
| `appearance_features.py` | Pure per-photo geometry ratios, semantic-region colour and pupil-excluding iris sampling |
| `feature_evidence.py` | Validate input hashes/curation/transforms; retain versioned features and parser masks |
| `metahuman\feature_comparison.py` | Downstream photo/render diagnostics and optional native eye-profile proposal; never edits a Character |

`features` requires explicit observation reuse; `inspect-image` performs fresh
dense inference on one image. Both parse actual native-resolution RGB, not the
aligned analysis colours, and retain source coordinates. Each output preserves
appearance cohorts without cross-photo averaging. Source changes fail closed;
no/multiple detected faces and insufficient individual features abstain.

Focused units cover colour clipping/iris pupil rejection, coordinate restoration,
stale curation/source changes, unknown/duplicate face IDs, mid-run invalidation,
single-render inspection and native proposal dependencies. They use synthetic
images/fake models; no model is loaded during units.

## Research decisions in plain language

| Decision | Reason / limit |
|---|---|
| Separate recognition, quality and anatomy | A high identity cosine can still describe a blurred or unusable crop |
| Complete-link grouping plus separate subject association | Reduce identity mixing without forcing every pose into one cluster |
| Keep native and aligned images separate | Centering helps analysis; warped pixels would corrupt reconstruction cameras |
| Semantic head mask plus soft matting | Person segmentation intentionally retains clothing |
| Bound colour changes and abstain | Scene neutrality/relative skin observations are not calibrated skin reflectance |
| Store derived images once | Reduce storage without replaying inference or changing original pixels |

Model sources and method tradeoffs: [research](research.md).

## Validation

```powershell
python -m unittest discover -s src\ailias\imageprocessing\tests -t src -v
```

Units cover decoding/EXIF, corrupt/duplicate input, grouping ambiguity,
co-occurrence, curation, unchanged pixels, source integrity, reversible alignment,
colour abstention, mask forwarding and cleanup. Fake backends replace models;
no user photos, network or GPU probes run.

Keep real model execution explicit and separate from unit discovery.

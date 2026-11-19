[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Imageprocessing internals

## Simple API

`images.py` owns one `ImageProcessor` and the plain `Frame` data record.
MODNet predicts opacity; BiSeNet labels a grey composite of that result and
restricts its alpha. Optional white balance completes preparation.
`face_feature_extraction.py` owns `FaceFeatureExtractor`: MediaPipe inference,
coordinate mapping and calls to `photo_features()`, not orientation or selection.
It consumes existing `image`, `original_image` and `regions` values on one
validated pixel grid; it never reloads source files, mattes, reparses or isolates
the head again. Neither class imports the other or depends on native-engine modules.
Each function has a short descriptive docstring naming its model where applicable.

`face_grouping.py` derives orientation from raw transforms, forms view groups,
then annotates quality and selects every usable extraction within each group.
It performs no inference, identity matching or native-engine operations.

The public entry points only orchestrate their scoped stages. Networks and
the dense detector load lazily, are reused serially, and have independent
`close()`/context lifecycles. Preparation outputs remain usable after the
processor closes. White balance changes only the prepared cutout; feature colours
are sampled from original RGB using existing semantic labels and dense points.
This preserves the uncorrected-photo colour contract, including its lighting/cast,
without retaining manifests. MediaPipe receives the prepared head, not original RGB.
`extract_features()` returns a facial-data-only `FaceFeatures`: raw predictions,
regional appearance, ratios, iris evidence and colour-source semantics.
The batch contract intentionally changed to `FaceExtractionResult`: its `.features`
contains that data while source/frame/backend/errors live alongside it.
The wrapper retains original/prepared pixels by reference for later grouping and
review export. Callers own their lifetime; no original is modified.
Missing/ambiguous detections and invalid points yield partial BiSeNet data without
landmark geometry; runtime failures are per-item batch errors. Direct inference/runtime
failures still raise. Consumers use attributes; preparation outputs remain dictionaries.
Output records live in model-free `face_features.py`; existing imports through
`face_feature_extraction.py` remain valid. `FaceFeatures.bisenet` contains
`FaceFeaturesBiseNet` with typed `FaceRegionFeatures` for all 12 natural regions
and the original combined eyebrow/lip summaries. `extract_bisenet_features()`
samples existing labels once and reuses that record in `photo_features()`;
no additional BiSeNet inference occurs. BiSeNet evidence is retained when
MediaPipe reports no/multiple faces or invalid landmarks, without inventing facial geometry.
Absent or clipped parts retain counts, parser IDs and available bounds with an
explicit review reason. Eye masks do not replace separate landmark-based iris sampling.
The legacy feature dictionary keeps its original keys and adds the individual
regions. `to_dict()` recursively produces plain data; `__str__()` emits strict
JSON, including stringified source paths, without writing artifacts.
The new class is retained output, not a new native-model integration. Existing
dictionary consumers use skin evidence for texture fill and hair/brow evidence
for approximate groom settings. Additional individual-region summaries currently
have no dedicated native consumer; they do not drive MediaPipe or 3D shape fitting.
`read_oriented_image(..., preserve_alpha=True)` supports the new loader;
its default remains RGB for all existing callers.

`ImageProcessor` defaults to `device="cuda"` with explicit `gpu_index=0`.
MODNet/BiSeNet check CUDA availability and verify graph execution, failing if
OpenCV silently ran entirely on CPU. `device="cpu"` is the explicit CPU-only
mode. `FaceFeatureExtractor` also defaults to CUDA mode; MediaPipe uses CPU on Windows and requests its GPU delegate
elsewhere; the selection reason is in `detect_face_landmarks()`'s docstring.
This is the owner's explicit Windows exception, not a failed-GPU retry.
See [MediaPipe's delegate support](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/base_options.py).
Existing callers of `DenseDetector(model)` keep their CPU default; the new API
always supplies the selected delegate.

Results and batch errors stay in memory. Preparation `ready` only confirms a
nonempty head; it is not image selection. Compose `process_images()` with
`extract_features_batch()` and `group_face_images()` while producer contexts are open.
Preparation errors retain decoded frame pixels/timing or in-memory inputs for review.
Grouping retains all candidates in their view groups, so bound large video batches.
Individual region/iris review flags remain measurement-level abstentions.

Grouping removes positive uniform scale from MediaPipe's transform before validating
a proper rotation; shear, reflection, degenerate and nonfinite transforms produce an
`unknown` category and diagnostic, never exclusion by themselves.
See the [MediaPipe transform contract](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/libs/procrustes_solver.h).
Pitch/yaw/roll are approximate uncalibrated model-coordinate angles, not certified
anatomical direction/camera elevation. Combined side-and-up/down bins retain both
axes. Large angles and roll have no rejection limit.

`FaceGroupingPolicy` loads `config\face_grouping.json`, or a caller supplies a
validated policy explicitly. View boundaries and quality-only thresholds are
documented next to their definitions in `face_grouping.py`. Raw landmark visibility
records alpha greater than zero; grouping independently applies its stricter alpha
threshold for quality warnings. It reuses the existing pure `face_quality()` pixel
metrics, not identity helpers or legacy selection/ranking.
Blur, size, exposure, clipping, alpha support and expression warnings never exclude
usable candidates. Only extraction failures or invalid/missing/entirely-outside
facial coordinate evidence are excluded.

`review_directory` is explicit. Excluded source photos are byte-copied; excluded
video frames or in-memory images are PNG exports. Destinations use exclusive
creation, partial copies are cleaned up, originals are untouched and per-item
copy errors remain in the report. Selected images write nothing. Group/decision
records serialize with `to_dict()`/`str()`; no JSON sidecars are written.
`FaceFeatures` contains none of this collection policy or bookkeeping.
Legacy `ProcessingPolicy` configuration remains accepted and unchanged; the new
grouper does not read identity settings or the old angle-rejection thresholds.
Model input sizes, landmark counts and trained RGB normalization remain named
model constants, not adjustable grouping thresholds.
`Frame` sampling uses presentation timestamps with bounded lazy consumption,
never average FPS; missing/invalid timestamps and rotated inputs fail explicitly.
No weights, codecs, GPU probes, real photos or native editors run in its mocked unit tests.

The new API assumes one known person across all inputs. Identity embeddings,
person clustering and subject association are intentionally out of scope, not
missing prerequisites. `face_grouping.py` organizes pose/view groups, followed
by keep-all-usable selection within those groups. The exactly-one-detection
check guards against ambiguous model output, not against different identities.
Legacy CLI identity grouping, curation and selection remain unchanged by request.

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
| `images.py` | Image/frame I/O, explicit cropping, MODNet/BiSeNet preparation and bounded colour normalization; no landmark inference |
| `face_feature_extraction.py` | MediaPipe lifecycle, raw native-grid predictions and facial measurements from existing pixels/labels |
| `face_features.py` | Facial-only evidence and separate extraction provenance/pixel wrappers; in-memory dict/JSON serialization |
| `face_grouping.py` | Orientation, view groups, quality warnings, selection and explicit excluded-image review copies |
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

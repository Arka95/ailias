[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Imageprocessing internals

| File | Ownership |
|---|---|
| `settings.py` | Validated processing policy, independent of the existing intake policy |
| `models.py` | Immutable model revision/checksums, explicit downloads, provenance/license retention |
| `backend.py` | `Backend` protocol and lazy CPU OpenCV implementation |
| `cuda.py` | Existing CUDA OpenCV DNN, same model preprocessing, no CPU retry, post-forward graph verification |
| `features.py` | Crop geometry, face quality, complete-link grouping, component association, contact sheets |
| `assets.py` | Canonical generated PNGs, content hashing and cross-run reuse |
| `curation.py` | Explicit reviewed-gallery capture, compatible exemplar features, three-way decisions and source-held-out diagnostic |
| `subjects.py`, `views.py` | Generation's seed-subject association and approximate view selection; do not rewrite stored groups |
| `matting.py`, `faceparsing.py`, `cleansing.py` | MODNet opacity plus face-anchored head semantics, bounded colour correction, immutable derivatives and review gates |
| `cleanup.py` | Verified legacy migration, protected-run pruning and unused generated assets |
| `pipeline.py` | Source references, decoding, run manifests, exports, selection metadata |
| `cli.py` | `models`, `scan`, `cleanse`, `select`, `compare`, `cleanup`, `curate` command contracts |

Public processing boundary:
`scan(store, source, policy, backend, recursive=False, body=True, limit=None,
progress=None)`. The backend exposes `detect(RGB_image) -> list[Face]`,
`segment(RGB_image) -> full_size_L_mask` and a provenance object.
Callers making changes must acquire `Store.lock()`; the CLI does this.
Tests inject a fake backend and never load model weights.

YuNet analysis is capped at 1600 pixels on the longest side by default; native
face boxes/landmarks are recovered by separate x/y scaling. SFace receives
OpenCV's five-point alignment internally. PP-HumanSeg uses RGB input normalized
to [-1,1], 192x192 inference, resized two-class logits and argmax, then a
nearest-neighbor visible-foreground mask in the source coordinate system.
Component labeling is capped at 1024 pixels to bound memory.

Only detection/embedding/segmentation operators run native model inference.
Exports preserve the original photo projection; normalized pose crops are never
fed back as reconstruction evidence.

Grouping is deterministic agglomerative complete-link for fixed detections,
input bytes and policy. Co-occurrence exclusions propagate across merges.
Ambiguity is assessed after clustering; uncertain points never become additional
competing seed aliases. This avoids the initial greedy implementation's
singleton-cascade bug. Complete linkage can still over-split varied poses.
Pairwise storage is quadratic; merge searches are worst-case cubic. This is
intended for personal photo sets, not an unbounded collection.

Each run saves an atomic manifest after every source, then final groups.
An orderly failed run becomes `failed`; hard termination can leave `running`,
which `select` also refuses. No implicit resume, cached inference replay or
background service exists. Job finalization removes only unreferenced files
inside that job's own directory through `resources.cleanup_owned_files`, keeping
saved evidence and all other jobs. Run IDs and resolved paths
are validated; recursive scans exclude the alias's own output tree.

Sources are referenced by absolute path and SHA256, checked again after decoding;
they are never copied. PNG assets live once per alias, keyed by their encoded
content hash. Fresh scans always run inference; this is storage reuse, not replay.
Selection verifies original hashes and references existing generated assets.
Intake accepts a selection manifest as well as a photo folder, without cloning
either. Crop provenance refers to oriented pixels; users retain originals for
later camera calibration. It does not update intrinsics automatically:
for an unscaled crop `(x0,y0,...)`, principal point becomes `(cx-x0,cy-y0)`.

## Curated filtering

`capture_curation(store, run, policy, removed_are_rejected=True)` is an explicit
human-label operation under the alias lock. It merges reviewed labels/features,
verifies unchanged kept PNGs, and stores no image copies. Merely running `scan`
never changes training examples. Unreviewed files in `assets\review` are skipped
until moved to portraits or removed during an explicit review.

`load_filter` validates alias, model/revision, feature dimensions and kept-file
integrity before a new run is created. The scan records the profile SHA256.
Classification combines exact source/crop and content-hash decisions with nearest
normalized positive/negative SFace cosine scores and quality gates. Encoder bytes
are reused for publication, not encoded a second time for hashing. Explicitly
kept files are reused directly, preventing a concurrent manual deletion from
being silently recreated. A model/dimension/stale-reference error is terminal.
Exact reviewed source/crop matches need no PNG re-encoding; unseen crops are
encoded once so identical reviewed pixels are also recognized across changed
original metadata or filenames.

Rejected detections retain metadata but have no exported products and are
excluded from visual grouping. Unknown/conflicting candidates retain a portrait
under `assets\review` and are ineligible for selection. Old-run selection also
checks the current curated gallery, so deleted old portraits are not reopened.
Bodies/masks for rejected detections are not exported. Segmentation association
still considers all detected faces, preventing rejected neighboring people from
being accidentally assigned to an accepted person's body.

Cleanup reads `curation.json` as a reference root and includes the review asset
category. Frozen kept examples survive pruning of their original scan; negative
examples retain features/hashes without retaining deleted pixels.
No new dependency, model download, environment or external image request is used.

## Research decisions in plain language

Updated November 16, 2026. Selection/cleansing additions do not establish a
successful real-head reconstruction. Proposed dense analysis is
not installed. This module owns image observations, not the fitted head.

### 1. Problem Statement

"Does this look like the same person?", "is this a useful crop?" and "what shape
is their nose?" are different questions. SFace supplies an **embedding**, a list
of numbers for recognition, not anatomical measurements. YuNet supplies five
image points, not eye contours, ears or a detailed face surface.

### 2. Implementation

| Decision | Plain-language reason and tradeoff | State |
|---|---|---|
| Learn explicit keep/delete decisions; quarantine uncertain new crops | A blurry crop can still strongly resemble the right person. Avoid deleting useful unfamiliar angles merely because recognition is uncertain | Implemented in `curation.py`; thresholds remain heuristics |
| Keep conservative stored groups; associate across them separately | Requiring every pair to agree reduces accidental identity mixing but can split one person across poses. Fixed seed examples allow broader selection without a chain of increasingly weak matches | Implemented in `subjects.py:associate_subject`; two independent seed-photo supports and co-occurrence exclusions |
| Estimate view independently of identity | Five points plus a generic face proxy estimate camera direction. Trying several focal lengths exposes some ambiguity; uncertain fits remain "unknown" | Implemented in `views.py:estimate_view`; approximate, not calibrated anatomy |
| Prefer frontal coverage without throwing away sides | Fronts anchor layout; other angles provide complementary observations. Current selection weights favour fronts, not a claim that fronts are best for every feature | Implemented in `subjects.py:select_views`; coverage and ranking heuristics, not learned accuracy |
| Add dense landmarks and expression/visibility assessment next | MediaPipe offers 478 facial landmarks and expression outputs. This is richer evidence, but predicted relative depth is not a measured scan, and dense face landmarks do not solve the whole head | Proposed; no MediaPipe runtime/model adopted |
| Add conservative white-balance/exposure harmonization | ICC-to-sRGB conversion standardizes colour interpretation, not illumination. Estimate a cast from scene-neutral evidence; never make skin itself grey or force every person to an average skin tone | Implemented in `cleansing.py`; bounded gains and subject-relative exposure, explicit skips, separate derivatives |
| Extract head-only foreground using semantics and opacity | Person matting deliberately keeps shirts. **Semantic labels** distinguish face/hair/ears from neck/clothes; MODNet then preserves soft outer hair edges | Implemented in `faceparsing.py` and `matting.py`, CUDA/explicit CPU; separate RGBA assets, unchanged review crops |

Why not blindly colour-neutralize every crop? Gray-world white balance assumes
average scene colour is grey; applying that assumption to a skin-dominated crop
can mistake real skin colour for a lighting cast. White balance also does not
remove spatial shadows or recover lighting-free albedo. Keep full-scene
illumination evidence before background removal, use conservative corrections
with review/abstention, and retain source RGB alongside any proposed derivative.

The cleansing contract includes **both colour consistency and head-only
background/clothing removal**, not just cropping. Preserve approved portrait bytes and
curation hashes; generate separately versioned corrected RGB, transparent RGBA
and aligned alpha/mask products without changing their pixel geometry. Keep
other people out of the selected subject; ambiguous/touching people and missing
or occluded anatomy need review, not fabricated foreground. Do not promise
perfect hair/glasses edges on every photograph.

### 3. Success Criteria

Preserve human-reviewed exclusions, unchanged originals and fixed seed identity
evidence. Audit chosen/review contact sheets and view coverage; **more selected
images is not proof of more accurate reconstruction**. Validate future dense
landmarks against visible features before using them to deform a head.
For cleansing, inspect alpha edges against light/dark/checkerboard backgrounds,
check hair/ears/beard are retained, and verify background suppression in the
downstream pipeline, not merely the existence of a transparent PNG. Check colour
correction against known-reference fixtures and expose uncertain corrections.

### 4. Task Breakdown

Next: improve matting/colour confidence calibration and review controls, then
dense observations with source coordinates, expression/occlusion
handling and per-region visibility. Hand observations/masks to generation;
leave dimensions/error reporting to [geometry](../geometry/dev.readme.md).

### 5. Reference Links

Current code: `features.py:group_faces`, `curation.py:load_filter`,
`subjects.py:associate_subject` / `select_views`, `views.py:estimate_view`.
See [existing model/curation evidence](research.md),
[multi-view decisions and primary papers](../generation/selection-research.md),
and [MediaPipe outputs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
Colour/background code: `pipeline.py:read_image` / `_scan`,
`backend.py:OpenCVBackend.segment`, `features.py:subject_bodies`.
Research: [OpenCV GrayworldWB and its assumption](https://docs.opencv.org/4.13.0/d7/d71/classcv_1_1xphoto_1_1GrayworldWB.html),
[scene context for skin/albedo estimation](https://arxiv.org/abs/2205.03962),
[MODNet portrait matting](https://arxiv.org/abs/2011.11961).

Implementation details: the official dynamic ONNX export fails OpenCV shape
import. Explicit model setup specializes it to `[1,3,512,512]` with ONNX Simplifier
and checks source/output equivalence on a generated input. ONNX/ONNX Runtime/
Simplifier are optional preparation tools, not the image-inference backend;
runtime matting stays on requested OpenCV CPU/CUDA. The original weights are
removed after successful preparation. Provenance retains original SHA256,
upstream revision, conversion recipe/tool versions and prepared model hash.
CUDA requires a completed forward plus backend graph evidence; no CPU retry.

The face parser is the pinned `yakhyo/face-parsing` ResNet18 BiSeNet ONNX
release (53,205,364 bytes; upstream MIT repository license retained).
Its 19 labels include neck/clothes as well as face parts. Runtime uses official
RGB/ImageNet normalization, argmax and nearest-neighbour label restoration.
The upstream repository license is recorded; training-data/commercial suitability
has not been established by this prototype.

Passing the whole tall head/shoulder crop initially still mislabelled neck/shirts
in several actual photos. Parsing now focuses on detector-anchored head context:
0.7 face-heights above the box, 0.1 below, and half-width at least 0.9 face-heights.
This is an inference crop, not a deformation or replacement mask; restored labels
retain native coordinates. Keep the connected head with most face-core overlap.
Only hair receives a narrow external fringe, intersected with MODNet opacity;
explicit neck/clothes/hat/jewellery labels are never restored by fringe growth.
If the semantic head reaches the context's lower boundary, flag review rather
than publish an apparently complete head clipped across the neck. These remain
heuristics: hair accessories, occlusions and ambiguous boundaries need review.

Primary implementation/reference:
[pinned inference](https://github.com/yakhyo/face-parsing/blob/4f0632b17a3e729a7871683aab7c89b6a595da9f/onnx_inference.py),
[class map](https://github.com/yakhyo/face-parsing/blob/4f0632b17a3e729a7871683aab7c89b6a595da9f/utils/common.py).

Cleansing manifests are independent of extraction manifests. Only completed
jobs advance the run pointer; failed jobs preserve recorded evidence and do not
replace a prior successful pointer. A changed curation fingerprint requires a
fresh cleansing run before generation. Cleanup follows these manifests as
reference roots, protecting originals, approved crops and shared derivatives.

## Validation

```powershell
python -m unittest discover -s src\ailias -t src -v
```

Fixtures cover duplicate/corrupt/no-face input, native crop pixels, EXIF/HEIC,
source preservation, body ambiguity, grouping bridge/tie/co-occurrence behavior,
quality exclusion, fresh inference with asset reuse, external source integrity,
zero-copy selection/intake, legacy migration and protected-run pruning.
Curated-filter units cover keep/delete labels, unseen similarity, conflict and
unknown abstention, original metadata changes, no automatic self-training,
manual approval, model/dimension failures and concurrent manual deletion.
Real-directory runs are separate opt-in CLI operations, not part of unit discovery.
The initial implementation was also exercised on the owner's requested folder;
see the current validation record in [research](research.md).

Head-only validation on November 16, 2026:

| Check | Evidence |
|---|---|
| Real curated run | Cleansing job `20261116T134753Z-c7f222a0`: 179 processed, 165 ready, 14 review; 3 multi-face crops excluded before processing |
| Review is not deletion | 12 masks reached the lower context boundary; 2 failed foreground/face coverage. All 182 reviewed portraits and the curation fingerprint remained unchanged |
| GPU execution | Completed OpenCV forwards: 95 CUDA/CUDA face-parser layers; MODNet 242 CUDA/CUDA plus one reported OCV/CUDA entry; RTX 4070 SUPER, no CPU retry |
| Colour abstention | 27 scene-neutral estimates, 148 ambiguous/excessive casts skipped, 4 insufficient-reference skips; exposure 70 harmonized, 40 clipping skips, 69 without references |
| Downstream proof | 64 photos from 18 groups; every RGB input verified as a hardlink and every binary native mask checked against published head alpha; all 495 products and their source/crop hashes verified |
| Functional regression | 143 isolated units passed, including head/ear/hair preservation, shirt/neck removal, soft hair fringe, missing/malformed labels, lower-boundary leakage review, native mask forwarding and immutable curation |

The ready preview was inspected: shoulders/shirts from the person-matting trial
are removed. This is not a guarantee of perfect accessory/ear/hair boundaries;
some hats/equipment can still be mislabelled. No new native head reconstruction
or personalized template fit was run for this cleansing milestone.

## CUDA backend

The owner confirmed reboot readiness on November 16, 2026. `--device cuda` chooses
`OpenCVCudaBackend` rather than pretending the shared headless CPU wheel supports
CUDA. The existing Thea `.venv` was verified to have OpenCV 4.13.0.90 with CUDA
13.1/cuDNN build support and CUDA DNN targets on the RTX 4070 SUPER.

`scripts\setup-gpu.ps1 -SourcePython PATH` reuses one existing environment through
the canonical `C:\projectWorkspace\.venv` junction, without copying files or
creating a venv. If the canonical interpreter already exists, that environment
is used directly and `-SourcePython` is optional. CUDA is validated before
installing ailias plus missing lightweight dependencies into the shared environment.
The target currently remains `Thea\.venv`, preserving existing callers.
No ailias-local environment or `.pth` overlay remains.
Do not install the CPU `imageprocessing` extra here: its headless OpenCV wheel
would shadow/replace the shared CUDA build.

The initially attempted redundant ONNX Runtime CUDA installation was stopped;
that backend/dependency was removed. It is not part of ailias's implementation.
The source environment already contains an unrelated torchvision metadata
dependency warning about missing torch; no such stack is needed or modified here.

YuNet inputs are padded to multiples of 32 and decoded using upstream model
conventions. OpenCV accepts variable spatial sizes without modifying the model.
SFace uses five-point similarity fitting to its 112x112 template; exports never
use the aligned recognition crop. PP-HumanSeg preprocessing matches the CPU path.

Networks explicitly request `DNN_BACKEND_CUDA` / `DNN_TARGET_CUDA` on device 0.
After a model's first successful forward, `Net.dump()` must show CUDA layers;
a pure CPU graph is refused. Per-backend/target layer counts disclose mixed
graphs rather than labelling all work GPU. An inference error is terminal,
not a per-file skip masquerading as GPU success. A model never called because
no face was detected is reported `not_executed`.

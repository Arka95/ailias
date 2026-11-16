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
| `cleanup.py` | Verified legacy migration, protected-run pruning and unused generated assets |
| `pipeline.py` | Source references, decoding, run manifests, exports, selection metadata |
| `cli.py` | `models`, `scan`, `select`, `compare`, `cleanup`, `curate` command contracts |

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

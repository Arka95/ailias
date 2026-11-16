[Back to REPO_ROOT](../../README.md)

# Imageprocessing

Responsibility: turn a file/folder into **reviewable, grouped portrait and visible
body candidates**, without altering original photos or inventing facial detail.
This module does not construct a MetaHuman, certify identity, infer neutral
expression, or replace capture review.

## Commands

```powershell
& C:\projectWorkspace\.venv\Scripts\Activate.ps1
python -m ailias imageprocessing models arka
python -m ailias imageprocessing scan arka C:\Users\bhowmikarka\Downloads\arka
python -m ailias imageprocessing scan arka C:\photos --recursive --limit 10 --no-body
python -m ailias imageprocessing select arka RUN_ID face-001
```

`models` explicitly downloads about 45 MB of SHA256-pinned YuNet, SFace and
PP-HumanSeg weights plus licenses into `data\arka\imageprocessing\models`.
Use `--no-body` on both model installation and scan to omit segmentation.
`scan` never downloads models. Image inference is local; no image,
embedding or crop is sent to a service.

### Explicit NVIDIA GPU execution

```powershell
.\scripts\setup-gpu.ps1 -SourcePython C:\projectWorkspace\Thea\.venv\Scripts\python.exe
& C:\projectWorkspace\.venv\Scripts\python.exe -m ailias imageprocessing scan arka C:\Users\bhowmikarka\Downloads\arka --device cuda
```

All applications use `C:\projectWorkspace\.venv`. This is a directory junction
to the existing Thea environment: **one physical environment, no clone or overlay**.
Setup installs ailias and missing lightweight dependencies into that shared
environment. Keep its existing target in place. No CUDA Toolkit, cuDNN or ONNX
Runtime download is required. Do not install the CPU `imageprocessing` extra
into this environment: its headless OpenCV wheel would replace/shadow CUDA OpenCV.

The default remains `--device cpu`; CUDA is opt-in, uses device 0, and refuses
unavailable CUDA or model-inference failure instead of retrying on CPU.
Run manifests include per-model backend/target graph verification after completed
inference, not just the GPU's presence. Models unused because there are no detections are
marked `not_executed`, not falsely GPU-verified. Some shape/transfer operators
can execute on CPU; decoding, image I/O, cropping and grouping are CPU work.
Mixed CPU/CUDA graph layers are reported honestly. Model handles are released
when the backend closes; source weights are never modified.

`scan` accepts one file or a directory. Scanning is nonrecursive unless requested.
Supported: JPEG, PNG, WebP, TIFF, HEIC/HEIF/HIF primary images. Exact byte duplicates
are analyzed once and retain all source filenames. Other formats are ignored;
unreadable supported files are reported without discarding successful images.
Each invocation performs fresh inference and creates an independent run.

The JSON response points to `manifest.json` and a group-overview JPEG. Open the
overview, then a group's contact sheets before running `select`. Group IDs are
run-scoped candidates, not person names; a large group can be the wrong subject.
The selection command is your explicit mapping of that group to the named alias.
Selection refuses superseded grouping revisions; old outputs remain available
for inspection, but require a new scan with the corrected grouping before export.

## Outputs under data\ALIAS

| Location beneath `data\ALIAS` | Contents |
|---|---|
| `imageprocessing\models\REVISION` | Verified weights, upstream licenses, model provenance |
| `imageprocessing\assets\portraits\SHA256.png` | Reusable native-resolution head/shoulder crops |
| `imageprocessing\assets\review\SHA256.png` | Uncertain new crops, separate from accepted portraits |
| `imageprocessing\assets\bodies\SHA256.png` | Reusable RGBA visible-person crops when association is unambiguous |
| `imageprocessing\assets\masks\SHA256.png` | Reusable corresponding binary masks |
| `imageprocessing\runs\RUN\groups\face-NNN` | Group-specific review contact sheets, not copied crops |
| `imageprocessing\runs\RUN\manifest.json` | Source links, EXIF correction, crop coordinates, features, filters and grouping |
| `imageprocessing\selections\SELECTION\selection.json` | Selected-candidate references and hashes, no copied images |
| `imageprocessing\selection.json` | Pointer/details of latest selection |
| `imageprocessing\comparisons` | Optional identity-versus-2D-landmark reports |
| `imageprocessing\curation.json` | Explicit human labels, frozen feature exemplars and validation diagnostic; no photo copies |

Original source folders are never written or cloned. Manifests keep absolute
source paths and hashes; moved, missing or changed originals fail integrity
checks rather than being silently replaced. Keep source photos available.
Repeated scans still run fresh inference, but identical generated PNGs reuse
the alias-level content-addressed asset. Crops are lossless PNG without
EXIF/XMP/GPS; originals retain their metadata. Source/crop coordinates are
pixels in the EXIF-corrected image. No crop is enlarged, rotated to flatten pose,
beautified, relit, denoised, sharpened or face-restored. ICC profiles are converted
to sRGB when valid; derived files are RGB8, so HDR originals must be retained for
any future radiometric/texture workflow.

## Filtering and grouping

`config\imageprocessing.json` controls face detector score, analysis resolution,
CPU threads, face-size/focus/exposure gates, crop margins and grouping thresholds.
Override with `scan --processing-config FILE.json`; no environment overrides.

- Quality uses the **face region**, not background detail: native face size,
  Laplacian variance after downsampling to at most 256 pixels, and exposure fractions.
- Portraits keep forehead/ear/jaw context. A crop containing another detected
  face is flagged and excluded from selection. Border clipping may still hide
  anatomy; human review is required.
- SFace embeddings use internally aligned faces **only for identity grouping**.
  Exported portraits keep native pose/projection.
- Conservative agglomerative complete-link cosine grouping requires similarity
  across **every** pair before merging groups. Co-occurring detections cannot
  share a group. Near-ties are flagged after clustering and excluded from
  selection, not used as competing anchors for later assignments.
- Every detection retains its features/decision in the manifest. With curation
  enabled, rejected portrait/body/mask files are not published; uncertain crops
  are separated for review. Selection excludes quality failures, uncertain
  identities and curation decisions other than accept.
- PP-HumanSeg supplies visible-person foreground, not pose/body reconstruction.
  Components are associated with face locations; a component containing multiple
  detected faces is marked ambiguous and no individual body is exported.

The default 0.50 cosine threshold is deliberately conservative, **not a calibrated
probability**. Viewpoint, glasses, lighting, mirrors, expression, lookalikes and
missed detections can split or mix groups. Similar-looking people require manual
inspection. Body masks are low-resolution model estimates, not hair-quality alpha
matting; connected/touching people may not be separable.

## Learn from manual portrait review

1. Finish reviewing a completed run's `assets\portraits` folder: retain valid crops and remove invalid ones.
2. Explicitly record that review with `curate`; missing files are treated as rejection **only in this command**, never inferred during an ordinary scan.
3. Run normal scans; the per-alias profile is loaded automatically before extraction.

```powershell
python -m ailias imageprocessing curate arka RUN_ID --removed-are-rejected
python -m ailias imageprocessing scan arka C:\photos --device cuda
```

The profile keeps each reviewed SFace feature vector rather than averaging all
poses/people into one centroid. Its labels mean **valid/invalid crops as reviewed
by you**, not independently certified person identities.

| Candidate | Pipeline behavior |
|---|---|
| Exact reviewed source/crop or pixel-content hash | Honor the human label; never regenerate a rejected portrait |
| Reviewed kept crop | Reuse its existing verified file; do not overwrite or recreate it |
| New crop close to kept examples with clear margin, passing quality checks | Save to `assets\portraits` |
| Strong rejected-example match, weak kept-example match and clear margin | Suppress portrait/body/mask exports; retain metadata and reason |
| Unknown, conflicting or quality-uncertain new crop | Save portrait to `assets\review`; selection remains blocked |

Defaults in `config\imageprocessing.json`:
`curation_accept_cosine=0.50`, `curation_reject_cosine=0.60`,
`curation_reject_positive_ceiling=0.35`, `curation_score_margin=0.10`.
New rejection requires negative score >=0.60, positive score <0.35 and
negative-minus-positive >=0.10. Acceptance requires positive >=0.50,
positive-minus-negative >=0.10 and no quality flags. These are conservative
engineering gates, **not probabilities or universally calibrated thresholds**.
Exact human labels take priority; an explicitly kept low-resolution crop is
preserved but can still fail reconstruction-quality/selection gates.

Review uncertain files separately. To approve one, move its unchanged
`SHA256.png` from `assets\review` to `assets\portraits`; to reject it, delete it.
Then run `curate` against that completed run with the confirmation flag.
The command merges explicit decisions with the existing profile, skips review
files still awaiting a decision, and does not learn from automatic predictions.
Subsequent manual deletions of a kept reference require another `curate` before
scanning; the pipeline fails closed rather than restoring that file. References
must remain unchanged, and the recognizer model/revision must match.

Capture also reports **leave-one-source-photo-out** accept/reject/review counts.
All crops from the held-out photo are excluded, and exact-match shortcuts are
disabled. This is a conservative diagnostic, not independent future-data accuracy.
Similarity alone cannot certify blur, anatomy, identity or reconstruction
readiness; same-person bad crops and unfamiliar valid poses often need review.
See the [research and actual validation](research.md#curated-portrait-filtering).

## Use selected candidates

The selection response provides `portraits`/`originals` path lists and a
`manifest` path. Pass that selection manifest to manual-first intake; it
references existing portraits without making another image directory:

```powershell
python -m ailias init arka
python -m ailias intake arka C:\projectWorkspace\ailias\data\arka\imageprocessing\selections\SELECTION\selection.json
```

Run `init` only if no session exists. Portraits may fail the main pipeline's
1024-pixel minimum even when the visible face passed this module's 256-pixel gate.
Never upscale just to pass: capture better photographs. Source originals and crop
transforms remain available for camera-aware reconstruction.

`scan` exit code 2 means one or more supported files failed; inspect `file_errors`.
Quality rejections and no-face images are normal report outcomes, not I/O errors.
`select` does not automatically ingest data or attest neutral expression.

Every scan finalizes a job-scoped cleanup pass. Only files within that new run
that are not referenced by its saved manifest are removed, along with empty
subdirectories. Failed runs are marked `failed` and keep recorded evidence for
diagnosis; uncommitted crop fragments are removed, including newly created
unreferenced shared assets. Selection failures and geometry-comparison failures
similarly remove their own uncommitted artifacts. Models, other jobs' referenced
assets and external photos are never swept. Hard process termination can
prevent finalizers.

Explicit maintenance is available:

```powershell
python -m ailias imageprocessing cleanup arka
python -m ailias imageprocessing cleanup arka --prune-superseded --keep-run RUN_ID
```

Cleanup migrates legacy source copies only when the original still exists and
its hash matches; otherwise it retains the only saved copy. It consolidates
legacy generated PNGs without temporarily duplicating their bytes, and removes
unreferenced managed assets. Pruning additionally removes obsolete complete/failed
runs, protecting the requested complete run and runs referenced by selections,
session/history or other reports. Models and original input folders are untouched.
`retained_empty_runs` reports empty directories Windows still holds open; their
image bytes have been removed, and a later cleanup retries directory removal.

For face comparison:

```powershell
python -m ailias imageprocessing compare arka RUN_ID FIRST_FACE_ID SECOND_FACE_ID
```

Face IDs/prefixes come from the run manifest, not group IDs. The response separates
identity cosine similarity from projected five-landmark shape error. It cannot
measure 3D nose projection, jaw shape, ears or hidden anatomy.

See [research](research.md), [internals](dev.readme.md), and
[geometry metrics](../geometry/README.md).

# ailias

Local, geometry-first photo-to-MetaHuman preparation with experimental native
reconstruction and editor adapters. **A real human head/MetaHuman has not yet been
produced or validated.** The source plan's manual-first end-to-end milestone
remains open; a successful tool call is not a likeness or production certificate.

Implemented: image intake, heuristic quality checks, manual angle/expression
review, nine-view coverage gate, durable stage handoffs, camera-pose records,
artifact integrity checks, downstream invalidation and three-view comparison sheets.
Optional **imageprocessing** adds local face grouping, portrait/body extraction,
HEIC support and face-specific quality filtering. **geometry** compares aligned
2D/3D landmarks and optional 3D surface samples.
**generation** runs installed CUDA COLMAP and Blender; the UE 5.8 adapter remains
blocked locally because Unreal is not installed. Native tools share one
workspace-level installation, not another Python/CUDA environment.

## Setup

Use **one shared workspace environment**, `C:\projectWorkspace\.venv`, for
workspace applications. It is a zero-copy directory junction to the existing
working Thea environment, not another installation. No app-local venv is created.
The setup script registers ailias and adds only missing lightweight packages;
it does not copy or replace the CUDA stack.

```powershell
Set-Location C:\projectWorkspace\ailias
.\scripts\setup-gpu.ps1 -SourcePython C:\projectWorkspace\Thea\.venv\Scripts\python.exe
& C:\projectWorkspace\.venv\Scripts\Activate.ps1
python -m ailias --help
python -m ailias init first-face
python -m ailias intake first-face C:\photos\reference
python -m ailias status first-face
```

Default session artifacts go in Git-ignored `data`. Configure thresholds in
`config\settings.json`; these are initial heuristics, not calibrated likeness
criteria. No network service or GPU work starts automatically. Only the explicit
`imageprocessing models` command downloads the pinned CPU analysis models.

## Clean and group a photo folder

```powershell
python -m ailias imageprocessing models arka
python -m ailias imageprocessing scan arka C:\Users\bhowmikarka\Downloads\arka
python -m ailias imageprocessing select arka RUN_ID face-001
```

Generated resources, model provenance, contact sheets, crops, masks and
selection metadata stay under **`data\arka`**. Originals remain in place and are
referenced by path and SHA256; generated crops/masks are shared across runs.
Inspect groups before selecting one;
largest does not necessarily mean the intended person. No face restoration,
warping or invented detail is applied. [Imageprocessing guide](docs/imageprocessing/README.md).

For GPU runs, use that same shared environment:

```powershell
& C:\projectWorkspace\.venv\Scripts\python.exe -m ailias imageprocessing scan arka C:\Users\bhowmikarka\Downloads\arka --device cuda
```

Jobs remove their own unreferenced temporary outputs, never other jobs' retained
results, original photos or models.
Explicit `imageprocessing cleanup` migrates verified legacy source copies and
reclaims unused generated assets. See the module guide before pruning old runs.

After manually reviewing/removing portrait crops, freeze those decisions:

```powershell
python -m ailias imageprocessing curate arka RUN_ID --removed-are-rejected
```

Later scans automatically honor that alias's reviewed profile. Known rejected
crops are not recreated; clear negative-exemplar matches are suppressed, while
uncertain new portraits go to `assets\review`, not the accepted portrait folder.
Existing kept portraits are never overwritten or recreated after another manual
deletion. [Curation rules and limitations](docs/imageprocessing/README.md#learn-from-manual-portrait-review).

## Usage

Use photo IDs from intake/status, with unique prefixes accepted by `review`.
Review must explicitly confirm or deny all three capture conditions.

```powershell
python -m ailias review first-face PHOTO_ID --yaw 0 --pitch 0 --neutral --unobstructed --consistent
python -m ailias validate first-face
```

`validate` returns exit code 2 until adequate accepted evidence exists. Complete
manual reconstruction/cleanup/MetaHuman work externally, then attach artifacts
and record the baseline using the [workflow guide](docs/workflow.md).

## Reconstruct and clean a candidate

```powershell
.\scripts\setup-generation-tools.ps1
python -m ailias generation doctor
python -m ailias generation reconstruct arka RUN_ID face-001
python -m ailias generation status arka JOB_ID
python -m ailias generation unreal-doctor
```

Only explicitly kept, quality-eligible portraits enter these experiments.
Reusable hardlinks avoid photo copies; jobs discard native databases/depth caches
after retaining their results or failure diagnostics. The current real-photo
benchmark is blocked by insufficient verified multi-view matches, not a missing
CUDA installation. Blender cleanup/rendering has run on a synthetic fixture;
that is **not** a reconstructed head. See [generation](docs/generation/README.md)
for gates, actual validation evidence and the remaining manual Identity solve.

| Document | Dev guide | Purpose |
|---|---|---|
| [Workflow](docs/workflow.md) | [Development](docs/dev.readme.md) | CLI contracts, capture and manual handoffs |
| [Imageprocessing](docs/imageprocessing/README.md) | [Module internals](docs/imageprocessing/dev.readme.md) | Portrait/body extraction and reviewable face aliases |
| [Geometry metrics](docs/geometry/README.md) | [Metric internals](docs/geometry/dev.readme.md) | Identity similarity versus actual geometric errors |
| [Generation](docs/generation/README.md) | [Native adapters and validation](docs/generation/dev.readme.md) | COLMAP, Blender, guarded UE 5.8 handoffs and honest failure states |
| [Implementation scope](docs/analysis.md) | [Remaining work](docs/todo/README.md) | Decisions, source and deferred functionality |

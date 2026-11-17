# ailias

Local, geometry-first photo-to-MetaHuman preparation with experimental native
reconstruction and editor adapters. **A photo-conditioned head baseline exists;
MetaHuman conversion and likeness remain unvalidated.** The source plan's manual-first end-to-end milestone
remains open; a successful tool call is not a likeness or production certificate.

Implemented: image intake, heuristic quality checks, manual angle/expression
review, nine-view coverage gate, durable stage handoffs, camera-pose records,
artifact integrity checks, downstream invalidation and three-view comparison sheets.
Optional **imageprocessing** adds local face grouping, portrait/body extraction,
HEIC support and face-specific quality filtering. **geometry** compares aligned
2D/3D landmarks and optional 3D surface samples.
**generation** runs installed CUDA COLMAP and Blender. **facefitting** fits a
complete GNM head using dense observations and batched CUDA optimization, then
adds photo-derived colour and explicitly approximate hair after clay approval.
A textured, upright static head is verified in the owned UE 5.8.4 review project;
a native head-only MetaHuman Character draft is also saved and reload-checked.
Final likeness, appearance transfer, rigging and assembly remain unvalidated.
Optional Creator Core Data is missing: skin editing is disabled, but local
Character creation and conform have succeeded without it.
See [Unreal readiness](docs/generation/README.md#unreal-explicit-installation-and-manual-identity-solve).
Native tools share one
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
`imageprocessing models` and `facefitting models` commands download their pinned models.

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
largest does not necessarily mean the intended person. Reviewed portraits and
native reconstruction pixels are not geometrically warped or face-restored.
[Imageprocessing guide](docs/imageprocessing/README.md).

Normal scans also create separate **head-only** cutouts (face, ears and hair,
not shirts/torso/neck) and
conservatively colour-harmonized native inputs; approved crops stay unchanged.
Cleansing also publishes separate centered, scale/roll-aligned analysis heads
with inverse source transforms, lighting evidence and complete paginated previews.
First prepare the optional conversion tools with
`python -m pip install -e ".[matting-tools]"`, then `imageprocessing models`.
To cleanse an existing reviewed run without detection/recognition again:

```powershell
python -m ailias imageprocessing cleanse arka RUN_ID --device cuda
```

Inspect the resulting alpha/colour preview. Missing colour evidence is explicitly
skipped and unreliable mattes require review; neither colour nor edge accuracy
is certified. See [cleansing behaviour and configuration](docs/imageprocessing/README.md#portrait-background-removal-and-colour-consistency).

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
CUDA installation. An explicit `config\generation-poc.json` profile relaxes
initialization/dense gates and has produced real photo-derived facial fragments
for Blender inspection; **not a complete head**. Default gates remain unchanged.
See [generation](docs/generation/README.md)
for gates, actual validation evidence and the remaining manual Identity solve.

For the separate coherent-head baseline, use
`python -m ailias facefitting fit arka RUN_ID face-001 --generation-config config\generation-poc.json`.
Its fresh 63-view CUDA trial took **42.187s versus 372.859s CPU** (8.84x faster),
with slightly better held-out landmark agreement. This is not a likeness score.
The separate appearance command now generates a textured Blender preview and
hair approximation without changing approved geometry.
See [renders, Blender stages, commands and limitations](docs/facefitting/README.md)
and the [researched UE 5.8 route](docs/generation/metahuman-workflow-investigation.md).

## Animate an existing MetaHuman

The separate [reanimation module](docs/reanimation/README.md) accepts narration
and an **already assembled rigged MetaHuman Blueprint**. It implements native
audio-driven face/head animation, a light-green portrait stage, GPU rendering
and explicit NVIDIA video encoding. **A personalized talking reel is not yet
validated: no assembled rig is available in the review project.** Audio can be
solved independently; this does not restart the stopped likeness experiments.

| Document | Dev guide | Purpose |
|---|---|---|
| [Workflow](docs/workflow.md) | [Development](docs/dev.readme.md) | CLI contracts, capture and manual handoffs |
| [Imageprocessing](docs/imageprocessing/README.md) | [Module internals](docs/imageprocessing/dev.readme.md) | Portrait/body extraction and reviewable face aliases |
| [Geometry metrics](docs/geometry/README.md) | [Metric internals](docs/geometry/dev.readme.md) | Identity similarity versus actual geometric errors |
| [Generation](docs/generation/README.md) | [Native adapters and validation](docs/generation/dev.readme.md) | COLMAP, Blender, guarded UE 5.8 handoffs and honest failure states |
| [Head fitting](docs/facefitting/README.md) | [Fit internals and measured evidence](docs/facefitting/dev.readme.md) | CUDA head fitting, photographic preview, approximate hair and verified static-head UE import |
| [Reanimation](docs/reanimation/README.md) | [Native adapter and validation](docs/reanimation/dev.readme.md) | Existing rig plus narration to GPU-rendered portrait reel; human output awaits rig validation |
| [Implementation scope](docs/analysis.md) | [Remaining work](docs/todo/README.md) | Decisions, source and deferred functionality |

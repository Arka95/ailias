[Back to REPO_ROOT](../../README.md) | [Developer guide](dev.readme.md)

# Native reconstruction and editor handoffs

This is an **experimental candidate-group benchmark**, separate from the strict
manual capture workflow. It never writes `session.json`, invents neutral-expression
reviews, confirms an identity, or marks a MetaHuman/likeness/animation as validated.
Casual photos taken at different times are not necessarily a rigid photogrammetry
capture. Blocked reconstruction must not be replaced with a generic placeholder.

## Setup and execution

Use the existing shared `C:\projectWorkspace\.venv`; no additional Python packages,
Torch, CUDA toolkit or application venv is needed for generation.

```powershell
Set-Location C:\projectWorkspace\ailias
.\scripts\setup-generation-tools.ps1
& C:\projectWorkspace\.venv\Scripts\python.exe -m ailias generation doctor
python -m ailias generation reconstruct arka RUN_ID face-001
python -m ailias generation status arka JOB_ID
```

The explicit setup command downloads SHA256-pinned official portable COLMAP 4.2.1
CUDA and Blender 4.5.15 LTS into `C:\projectWorkspace\tools`, once for workspace
applications. Archives are removed after extraction. Blender's bundled Python is
part of Blender, not another application venv. No engine, service or model is
downloaded by `doctor`, `status`, or a generation job.

`reconstruct` requires a completed imageprocessing run and an explicitly captured
curation profile. It takes up to 64 distinct photos from the seed-subject
hypothesis across supported groups, with approximate frontal preference and
complementary directions, excluding not-explicitly-kept crops. Candidate
identity ambiguity is recorded, not silently
converted into an identity/capture attestation.

The supplied generation profiles enable `use_cleansed_portraits`. `reconstruct`
first performs fresh CUDA head-only cleansing from `config\cleansing.json`,
without redoing detection/recognition or modifying approved crops. Native tools
receive corrected RGB with scene/clothes/neck zeroed and thresholded head-alpha
masks, not face ellipses. SIFT and foreground fusion explicitly receive the mask
directory; texturing receives the background-suppressed RGB. The pinned native
texturer has no separate mask option, so unsupported surface areas can still
receive black/inpainted texture; no per-texel visibility or true albedo is claimed.
Review/missing mattes are excluded; stale/changed derivative hashes fail closed.
Legacy benchmarks can explicitly set `use_cleansed_portraits=false`.

Audit prepared inputs without running models/native reconstruction:

```powershell
python -m ailias imageprocessing cleanse arka RUN_ID --group face-001 --device cuda
python -m ailias generation select-inputs arka RUN_ID face-001
```

The audit includes selected/review contact sheets and selection provenance.
See [image cleansing details](../imageprocessing/README.md#portrait-background-removal-and-colour-consistency)
and [research decisions](selection-research.md).

The job runs GPU SIFT extraction/matching with aligned head masks (face ellipses
only for the explicit legacy opt-out), then checks the
verified view graph before running SfM, CUDA stereo, foreground fusion, Poisson
meshing and photograph-atlas texturing. Recovered cameras must meet registration,
point-count and measured reprojection gates. CPU mapping/meshing is intentional;
GPU feature/stereo/render stages do not silently fall back to CPU.

Settings live in `config\generation.json`; override with `--generation-config FILE`.
Native dependency URLs/hashes live in `config\generation-tools.json`.
Exit codes: `0` requested operation/status read succeeded; `1` bad input/config/I/O;
`2` blocked/failed reconstruction, Unreal prerequisite failure, or status integrity
problems. A healthy status read can return `0` for an explicitly blocked job.

## Permissive first PoC

The stricter default is not required for an early visual experiment:

```powershell
python -m ailias generation reconstruct arka RUN_ID face-001 --generation-config config\generation-poc.json
```

This explicit profile permits at least three registered photos and 4% of the selected set,
20 sparse points, 8 px mean reprojection error, 50 fused points and 50 faces.
It lowers mapper initialization to 20 inliers/0.5-degree triangulation, absolute
pose support to 15 inliers, dense consistency to one supporting source view and
fusion to two pixels. Defaults retain the previous native 100/16-degree/30/2/5
settings. It can expose fragments rather than blocking early, but does not certify
camera accuracy, complete anatomy or likeness. It does not relax portrait
curation, original-file integrity or cleanup isolation.
It allows explicitly kept soft blur/exposure/size warnings, but excludes faces
below 64 pixels and crops flagged as containing another face. Dense side is
1,200 and Poisson depth 10; these cannot manufacture observed facial detail.

On November 16, job `20261116T122845Z-ed50db97` produced real photo-derived
geometry: 3 registered photos, 4,263 fused samples and 6,164 textured faces.
Inspection shows **13 disconnected fragments, not a usable full head**; the
90-degree diagnostic is mostly empty. A completed software job is not a
completed person. Its local editable scene is:

```text
data\arka\generation\runs\20261116T122845Z-ed50db97\blender\head.blend
```

An earlier, sparser PoC scene was opened successfully in the owner's installed
Blender 5.3. Native generation still uses the shared, tested Blender 4.5 installation.
Save manual edits as a new file to preserve the hashed generated revision;
in-place saves are detected as artifact changes, not silently accepted.

## Artifacts and cleanup

Each alias uses `generation\inputs\CONTENT_HASH` for reusable native image views
and masks. Images are hardlinks to existing generated portraits, **not copies of
originals**. Failure to create a same-volume hardlink is an error, not permission
to silently clone photos. An unchanged input set reuses the same view/masks.

Jobs live in `generation\runs\JOB_ID`. Finalizers retain only the manifest,
bounded native logs, validated reconstruction products and committed editor
outputs. Databases, temporary model exports, undistorted images and depth/normal
caches are removed on success/failure; cleanup errors are reported. Canonical
portraits, sources, models, other jobs and external Unreal projects are never
deleted. Imageprocessing cleanup honors generation references.

Successful reconstruction retains a textured OBJ/material, one photograph atlas,
raw sparse model text, recovered camera JSON and, after successful meshing, the
raw fused observed surface. Cameras preserve unknown SfM scale and restore
portrait crop offsets for full-source intrinsics. These are
**not** calibrated metric cameras and cannot be attached to the manual workflow
without independently establishing scale/conventions and accepted-photo coverage.

Blender removes zero-area faces/loose vertices, recalculates consistent normals,
preserves surface coordinates/UV seams, and exports an OBJ/material plus `.blend`.
It does not smooth, invent missing detail, fill holes, remove disconnected ears,
orient/scale a human head, or certify watertightness. One recovered-camera render
and 45/90-degree diagnostic camera orbits are retained. The orbit labels are not
human-verified frontal/profile views or matched-camera likeness comparisons.

If reconstruction products exist but Blender failed:

```powershell
python -m ailias generation clean-mesh arka JOB_ID
```

Committed cleanup outputs are immutable revisions, not overwritten by retries.

## Unreal: explicit installation and manual Identity solve

UE 5.8.4 and the MetaHuman Character plugin are installed. A real NullRHI
commandlet loaded the head-conforming Python APIs in the owned
`data\arka\unreal\AiliasReview\AiliasReview.uproject`; it did not create assets or
request cloud work. Actual import/conform is still unverified.
**MetaHuman Creator Core Data is missing**: the editor logged its limited-features
warning, and its Optional texture-model/body-texture folders are absent.
Install that component through Epic Games Launcher -> UE 5.8 -> Options ->
MetaHuman Creator Core Data -> Apply, with projects closed.
[Epic installation instructions](https://dev.epicgames.com/documentation/metahuman/getting-started-with-metahuman-creator-in-unreal-engine?application_version=5.8).

Use a project with PythonScriptPlugin and MetaHumanCharacter enabled.
The adapter checks engine/project/plugin metadata and
refuses a mismatched numeric engine association rather than silently upgrading it.
`unreal-doctor` also reports `metahuman_core_data` and warnings. Its existing
`ready` field covers engine/project/plugins, not complete content, an executed
solve or likeness; missing skin content does not block plain static-mesh import.
No Epic credentials are collected or stored by ailias.
The commands below target native generation jobs and an already-solved Identity,
not the new GNM fitting jobs. For arbitrary head topology, the researched next
experiment is [UE 5.8 head-only Custom Mesh](metahuman-workflow-investigation.md).

```powershell
python -m ailias generation unreal-doctor `
  --editor 'C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe' `
  --project C:\UnrealProjects\Ailias\Ailias.uproject
python -m ailias generation unreal-import arka JOB_ID `
  --editor 'C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe' `
  --project C:\UnrealProjects\Ailias\Ailias.uproject
```

Import creates a textured static mesh under a job-specific `/Game/Ailias/...`
namespace without overwriting existing assets. This is **not MetaHuman creation**.
Actual `.uasset`/bulk outputs are hashed, not replaced by a success receipt alone.

In the editor, inspect geometry, scale, open eyes and material quality. Follow
Epic's **From Mesh** workflow: create an Identity from the mesh, promote/track a
neutral view, manually inspect markers, solve, and conform through the supported
Identity workflow. Enable the engine's Identity/MetaHuman Animator support as
required. Create a MetaHuman Character asset before invoking:

```powershell
python -m ailias generation unreal-conform arka JOB_ID `
  --editor 'C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe' `
  --project C:\UnrealProjects\Ailias\Ailias.uproject `
  --identity /Game/Characters/ArkaIdentity --character /Game/Characters/Arka
```

The adapter checks `ImportErrorCode.SUCCESS`, commits/saves face state and
releases its editor registration. It never passes arbitrary topology to the
MetaHuman-template import API or assumes measured scale.

Only adding **`--cloud-services`** requests Epic's blocking auto-rig/texture
services. First authenticate interactively through Epic's editor with MetaHuman
Cloud access; commandlets cannot substitute for that account session. Cloud
requests returning does not certify a usable facial rig.

**Local UE execution, cloud authentication, assembly/DCC export and expression
validation remain unverified/unimplemented as applicable.** The adapter does not
claim assembly or rig/likeness validation. Editor failures may leave partial
assets in the external project; receipts record that, and ailias does not delete
or pretend to roll back those assets.

See the [native validation record and primary sources](dev.readme.md#validation).

[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Facefitting

Fit one complete GNM head to multiple photographs, keeping identity separate
from each view's camera and expression. Generate Blender geometry/appearance
previews and hand a hair-free mesh to Unreal.

**Not complete:** automated native resemblance refinement, photographic skin
transfer, groom binding and a reusable rig/assembly pipeline.

## Inputs and outputs

| Operation | Required input | Retained output / next consumer |
|---|---|---|
| `fit` | Curated imageprocessing run/group, current cleansing, pinned GNM/landmarker assets | Neutral OBJ, parameters, observations, cameras, overlays and review scene |
| `appearance` | Exact fitted mesh plus matching geometry-review record | Photo-colour atlas/coverage, UV geometry, approximate hair, scene and hair-free OBJ/MTL |
| `unreal-import` | Verified appearance revision, UE editor/project | Textured static mesh/material/texture packages; not a MetaHuman |
| `unreal-conform` | Verified import and optional pinned baseline | New head-only Character, tracked contours, camera and saved-package evidence |

Jobs are retained under `data\ALIAS\facefitting\runs`, `appearance`, `unreal`
and `conforms`. A successful conform transfers geometric constraints, **not**
the Blender skin atlas, eye appearance or facial/scalp hair.

## Usage

```powershell
python -m ailias facefitting models subject
python -m ailias facefitting fit subject RUN_ID face-001
python -m ailias facefitting status subject FIT_JOB
python -m ailias facefitting appearance subject FIT_JOB
python -m ailias facefitting appearance-status subject APPEARANCE_REVISION
$ue = 'C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
$project = 'C:\path\to\Review.uproject'
python -m ailias facefitting unreal-import subject APPEARANCE_REVISION --editor $ue --project $project
python -m ailias facefitting unreal-conform subject IMPORT_JOB --editor $ue --project $project
```

`appearance` requires `reviews\FIT_JOB-clay-approval.json` with decision
`approved`, scope `neutral_clay_geometry`, and the matching mesh reference.
This is a technical dependency gate, not evidence that texture or final likeness passed.

## Configuration

| Selector | Default | Scope |
|---|---|---|
| `fit --fit-config FILE` | `config\facefitting.json` | Optimizer, shape modes and regularization |
| `appearance --appearance-config FILE` | `config\appearance.json` | Texture/hair preparation budgets |
| `--generation-config FILE` | `config\generation.json` | Shared native tools, Blender device/render size and timeout; supported by fit, appearance, import and conform |
| `unreal-conform --baseline FILE` | Alias `facefitting\working-baseline.json`, when present | Pinned existing scene/geometry/camera dependencies |
| Model specifications | `config\facefitting-models.json` | Source URLs, revisions, checksums and retained licenses; no CLI path override |

### Fitting: `config\facefitting.json`

| Key | Default | Meaning |
|---|---|---|
| `device` | `cuda:0` | Explicit CUDA index; `cpu` selects the SciPy reference, with no silent fallback |
| `max_iterations` | `800` | CUDA optimizer iteration budget |
| `identity_candidates` | `[48, 96]` | Shared identity mode counts compared on held-out views |
| `expression_modes` | `[12, 12, 24]` | Retained mode counts for the three model expression blocks |
| `identity_regularization` | `0.00001` | Penalty on identity coefficients |
| `expression_regularization` | `0.000005` | Penalty on per-photo expression coefficients |
| `max_nfev` | `55` | CPU least-squares evaluation budget |

The JSON profile selects CUDA; constructing `FitPolicy()` directly defaults to CPU.
Model specifications use `file`, `url`, `sha256` and `limit` for cache identity,
download integrity and byte limits; `license`/`source` retain provenance.
GNM, MediaPipe, the surface correspondence and their license/readme records are pinned separately.

### Appearance: `config\appearance.json`

| Key | Default | Meaning |
|---|---|---|
| `device` | `cuda:0` | Explicit CUDA device for photo projection |
| `texture_size` | `2048` | Square colour-atlas side in pixels |
| `photo_size` | `1536` | Maximum prepared photo side for projection |
| `max_views` | `16` | Maximum selected texture observations |
| `render_samples` | `24` | Cycles appearance-preview sample count |
| `hair_strands` | `14000` | Approximate procedural scalp-strand budget; `0` disables strands |
| `hair_segments` | `8` | Segment count per generated strand |
| `seed` | `42` | Deterministic procedural layout seed |

Shared native configuration is documented in [generation](../generation/README.md#configuration).
Texture colour still contains source illumination; it is not intrinsic albedo.
Unobserved texels are neutral-filled and excluded from measured coverage.

## Current working baseline

`data\arka\facefitting\working-baseline.json` pins the existing diagnostic scene,
source geometry/texture, observations and cameras. Preserve these files;
native refinement must not overwrite the baseline or silently regenerate a Blender fit.

The geometry reference is revision `20261116T153125Z-44390de9`; the appearance
reference is `20261116T163340Z-14973ca7`. Their roles differ: a fixed geometric
reference does not validate the appearance. See the [current comparison](../../README.md#current-state---below-expectations).

## Photo appearance preview

| Neutral clay | Photo-colour and approximate hair |
|---|---|
| ![Fitted geometry](../images/pipeline/clay.png) | ![Blender appearance preview](../images/pipeline/appearance.png) |

Generic eyes, texture seams, residual photographed lighting and inferred hidden
anatomy remain. The preview renderer also scales inherited light energies to
8%; its displayed skin tone is not a calibrated reference.

## Animated native draft

The existing baseline-backed Character has a full face rig and saved runtime
assembly in the local review project. An actual 743-frame, 540x960, 30 fps reel
passed media decoding and artifact checks. Its default/native appearance does
not reproduce the photo-derived Blender appearance or a faithful personal likeness.

`data\arka\facefitting\reviews\personal-runtime-character.json` records native
package and rig/assembly/reel references. It is technical provenance, not a
likeness certificate. Animation of an existing compatible actor belongs to
[reanimation](../reanimation/README.md), not another head fit.

## Native Character draft and hair later

The conform adapter uses a matched perspective tracking view and native
`(GNM X, Z, Y) * 100` target coordinates. Static import instead uses
`(GNM Z, -X, Y) * 100`; these conventions must not be mixed.
Nominal scaling is not physical calibration.

The exported Alembic is not a validated bound groom: the retained native import
reported twice the intended strand count, with axis/root alignment unresolved.
Adding a beard or copying a texture file alone does not establish a working
deformable appearance. See [handoff internals and tests](dev.readme.md).

[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Facefitting

Fit one complete GNM head to multiple photographs, keeping identity separate
from each view's camera and expression. Generate Blender geometry/appearance
previews and hand a hair-free mesh to Unreal.

**Not complete:** automatic photo-guided native resemblance fitting and
photographic skin transfer. Scripted feature profiles now persist native groom,
iris, bounded sculpt and texture-override edits; their visual quality remains a
separate check.

## Inputs and outputs

| Operation | Required input | Retained output / next consumer |
|---|---|---|
| `video-create` | Retained sampled-frame manifest, installed models, Blender and UE project | Fresh observations/fit, textured Blender scene and hair-free mesh handoff, native-UV colour, native grooms/iris, independent reload and new editor portrait; likeness remains unapproved |
| `fit` | Curated imageprocessing run/group, current cleansing, pinned GNM/landmarker assets | Neutral OBJ, parameters, observations, cameras, overlays and review scene |
| `appearance` | Exact fitted mesh plus matching geometry-review record | Photo-colour atlas/coverage, UV geometry, approximate hair, scene and hair-free OBJ/MTL |
| `unreal-import` | Verified appearance revision, UE editor/project | Textured static mesh/material/texture packages; not a MetaHuman |
| `unreal-conform` | Verified import and optional pinned baseline | New head-only Character, tracked contours, camera and saved-package evidence |
| `unreal-features` | Existing native Character, explicit feature profile, pinned baseline when present | Isolated Character revision, edit receipt and independent editor reload; optional runtime assembly |
| `unreal-review` | Verified feature revision with runtime Blueprint | One neutral native frame, saved stage/camera policy and source hashes; no audio solve |
| `native-control-map` | Existing native topology/export receipt, unchanged source package and installed landmark metadata | Exact API-index/canonical-vertex associations and available named curve memberships; no sculpt movements |

Jobs are retained under `data\ALIAS\facefitting\runs`, `appearance`, `unreal`
and `conforms`; native edits/reviews use `features` and `native-reviews`.
A successful conform transfers geometric constraints, **not**
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
python -m ailias facefitting unreal-features subject --character /Game/Heads/Person.Person --profile C:\profiles\features.json --editor $ue --project $project
python -m ailias facefitting unreal-features subject --character /Game/Heads/Person.Person --profile C:\profiles\features.json --assemble --editor $ue --project $project
python -m ailias facefitting unreal-review subject FEATURE_JOB --editor $ue --project $project --resolution 540x960
```

### Fresh run from sampled photos

```powershell
python -m ailias facefitting video-create subject --evidence C:\data\sampled-photos\manifest.json --editor $ue --project $project
```

`--evidence` reuses only the supplied sampled images and capture timestamps.
This command never decodes the original videos, resamples images or reuses
their previous landmark/expression/semantic predictions. It runs fresh
MediaPipe landmarks and regional face parsing on the retained images, then uses installed GNM
weights and fits a new shared identity/cameras/expressions using the existing
`config\facefitting.json` policy, imports a new mesh, creates a new native
Character with the factory, conforms it and independently reloads it. It does
not read a previous personal head, fit, native response, ray correspondence or
appearance profile. Before Unreal import, the command now creates a fresh
Blender clay scene, bakes photo colour onto the fitted GNM UVs, omits scalp hair
by default, saves `blender\textured-head.blend` and
exports the hair-free mesh that Unreal actually consumes. The Blender export
must preserve fitted vertices under the declared axis/centimetre conversion.
Missing scenes, requested hair, render outputs or failed Blender processes stop
the run; the command does not fall back to the direct Python OBJ.
No intermediate approval prompt or manual tuning is part of execution.

`--photos-run RUN` adds every accepted original-photo portrait across that
imageprocessing scan's groups; old eligibility and group-size thresholds do
not remove accepted portraits. Hash-verified originals retain their full image
coordinates and resolution; reviewed crops localize detection only. Fresh
landmarks are mapped back before camera fitting, and parsing is recomputed.
Review rejections, ambiguous
target detections and unsupported views remain explicitly accounted for.
The source-photo inventory is retained in `photo-inputs\manifest.json`.

`--fit-search-config config\facefitting-search.json` replaces the small
identity-only comparison with an explicit 162-candidate search:
3 identity capacities x 3 expression capacities x 3 identity-prior strengths x
3 independent expression-prior strengths x 2 expression bounds. The full
supported expression basis is 100/100/150 regional modes. Candidates are
ranked using view-category-balanced RMS over all supported held-out landmarks,
not only median residuals. Every candidate's parameters/report is retained;
failed candidates are visible rather than silently omitted. Camera/expression
are still optimized on validation images: this is identity validation, not an
independent 3D scan or proof of likeness.

The selected fit is refitted using all prepared images, produces a source-pixel
landmark overlay for each view under `fit-overlays`, and continues through the
same Blender/Unreal stages. `view_batch_size` controls GPU Jacobian memory only;
it never selects or discards images. Both options are opt-in because a broad
search and original-photo decoding are materially more expensive than a small
sampled-image run.

The completed original-photo/search run `252acd17` fitted 260 views and tested
all 162 configurations without a failed trial. It selected 96 identity modes
and the full 100/100/150 expression basis, then completed Blender and Unreal.
Its actual renders were inspected; likeness remains unapproved, and texture
seams, dark native lighting and approximate grooms remain visible.
See [measured results and remaining bounds](dev.readme.md#fitting-restrictions-and-search).

The command now connects native skin UV export, fresh photo-colour baking and
Texture2D import to native skin, iris, eyebrow, scalp-hair, facial-hair and
eyelash edits. `--appearance-config` uses `config\appearance.json` for the
CUDA device, atlas resolution, photo cap and view count. Stock groom selection,
pigment mapping and landmark-region facial-hair darkness are **approximations**;
the parser has no distinct beard/moustache class. Eyelashes use an explicitly
unmeasured preset. Observed colour retains photo lighting; unseen atlas texels
are labelled median-skin fill, not recovered detail.

The earlier direct-native command completed through native save/reload
and a new close-up image, but omitted Blender. Its delivery run
`b58c0c440faa4df5900e613ad8faa7a1` completed in **540.265 s** from 82
retained photos, with correctly oriented photo detail, actual native grooms,
corrected portrait framing and zero surface displacement on independent reload.
All 131 checked input/output artifact references still matched. This historical
result predates the package reorganization and required Blender stage; it is
not validation of the revised full pipeline. Its new image
was actually opened for review; older reversed/wide previews are not substituted.
The photo tint still includes source lighting, body appearance is an unchanged
generic default and no visual likeness acceptance is implied. Actual actors expose scalp
hair, brows, beard, moustache and lash groom assets.
`fresh_video_completed` means unattended editable-Character
creation plus a **new editor-native preview**, not an assembled animation rig
or accepted likeness. Any failed stage exits **2**. Results and per-stage/wall timings are under
`data\ALIAS\facefitting\fresh-video\REVISION\manifest.json`.
Wall time starts from the retained sampled photos, not from video decoding.
This path needs the existing facefitting, imageprocessing and explicitly
selected CUDA runtime prerequisites; it does not install or replace GPU stacks.
See [measured failures, fixes and limitations](video-identity-investigation.md).
The revised working-tree unit cohort passes **265 tests**, including Blender
failure/geometry handoff and photo-projection cleanup cases. Units use mocked
native tools; they are not proof of a real full run or of likeness.

The revised full public command subsequently passed as
`17b9ff17ee584c24b6d8f5d42e981ec5` in **466.375 seconds**, including Blender.
It saved a new `blender\textured-head.blend`, exported the actual Unreal input,
created 14,000 approximate Blender strands and produced a new native Character
and portrait. All 157 unique checked references match and native surface
max/RMS reload drift is zero. Both actual new Blender/native images were opened.
See [timings, outputs and visible limitations](dev.readme.md#verified-blender-to-unreal-run).

`appearance` requires `reviews\FIT_JOB-clay-approval.json` with decision
`approved`, scope `neutral_clay_geometry`, and the matching mesh reference.
This is a technical dependency gate, not evidence that texture or final likeness passed.

## Configuration

| Selector | Default | Scope |
|---|---|---|
| `fit --fit-config FILE` | `config\facefitting.json` | Optimizer, shape modes and regularization |
| `appearance --appearance-config FILE` | `config\appearance.json` | Texture/hair preparation budgets |
| `--generation-config FILE` | `config\generation.json` | Shared native tools, Blender device/render size and timeout; supported by video-create, fit, appearance, import and conform |
| `unreal-conform --baseline FILE` | Alias `facefitting\working-baseline.json`, when present | Pinned existing scene/geometry/camera dependencies |
| `unreal-conform --reference-face-id ID` | Off; existing Blender tracking preview | Track original pixels from one already-fitted photograph using its retained perspective camera; no new GNM fit or Blender render |
| `unreal-conform --source-character PATH` | Off; create a fresh draft | Duplicate and refine the existing native Character in a unique namespace; its source package is hash-pinned and never saved |
| `unreal-features --profile FILE` | Required | Validated native edit policy; no inferred anatomical correspondences |
| `unreal-features --baseline FILE` | Alias working baseline, when present | Verify every pinned source; never refit or rewrite its Blender scene |
| `unreal-features --assemble` | Off | Build using profile `assembly` settings (optimized/medium by default); independently reload and inspect the actor/rig/grooms |
| `unreal-features --cloud-services` | Off | Allow missing native full-rig/source-texture requests; existing compatible data is reused |
| `unreal-review --review-config FILE` | `config\reanimation.json` | Same documented camera, lights, placement, warmup and render policy as [reanimation](../reanimation/README.md#configuration) |
| `unreal-review --resolution` | `540x960` | Even 9:16 review dimensions; does not change the stored camera/light defaults |
| `unreal-review --diagnostic` | Off | Inspect retained runtime assets from a failed revision; does not mark its checks or likeness as passed |
| Model specifications | `config\facefitting-models.json` | Source URLs, revisions, checksums and retained licenses; no CLI path override |

### Fitting: `config\facefitting.json`

| Key | Default | Meaning |
|---|---|---|
| `device` | `cuda:0` | Explicit CUDA index; `cpu` selects the SciPy reference, with no silent fallback |
| `max_iterations` | `800` | CUDA optimizer iteration budget |
| `identity_candidates` | `[48, 96, 170]` | Compare coarse through full supported head-identity bases on held-out views; extra modes are not extra measured points |
| `identity_coefficient_limit` / `expression_coefficient_limit` | `3.0` / `3.0` | Symmetric coefficient bounds; the search also tests expression bounds of 6 |
| `focal_factor_min` / `focal_factor_max` | `0.1` / `5.0` | Focal pixels divided by source long side; replaces the binding 0.45 floor |
| `view_batch_size` | `16` | Views per CUDA Jacobian batch; every view contributes to each shared-identity iteration |
| `expression_modes` | `[12, 12, 24]` | Retained mode counts for the three model expression blocks |
| `identity_regularization` | `0.00001` | Penalty on identity coefficients |
| `expression_regularization` | `0.000005` | Penalty on per-photo expression coefficients |
| `max_nfev` | `55` | CPU least-squares evaluation budget |

The JSON profile selects CUDA; constructing `FitPolicy()` directly defaults to CPU.
Model specifications use `file`, `url`, `sha256` and `limit` for cache identity,
download integrity and byte limits; `license`/`source` retain provenance.
GNM, MediaPipe, the surface correspondence and their license/readme records are pinned separately.

### Photo-contour native refinement

```powershell
python -m ailias facefitting unreal-conform subject IMPORT_JOB --editor $ue --project $project --reference-face-id SOURCE_FACE_ID --source-character /Game/Heads/Person.Person
```

This reuses the approved source mesh, existing per-photo fitted camera and head
alpha. Actual decoded photo pixels are placed on a neutral **full-source-frame**
background, then resized with their aspect ratio intact; there is no aligned
cutout deformation, cropped principal point or new Blender scene. Camera position,
forward/up axes and horizontal field of view are derived from the retained
world-to-camera fit. The selected face ID must exist in the inputs, cameras and
dense observations, with matching source hashes/dimensions.

The existing native Character is duplicated before UE's head-only solve.
An inherited full face rig is removed on the **copy** before geometry editing;
subsequent assembly requires a fresh compatible rig. Already-rigged source
copies use D3D12/offscreen rather than NullRHI. This is a
new **native** solve, not a new GNM fit; the original photo, approved mesh and
selected Blender scene remain untouched. Existing camera estimates and photo
expression/occlusion can still be wrong. Choose a visible, near-neutral face;
these flags do not claim automatic reference selection or verified likeness.

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

### Native feature profile

The profile edits a duplicate of the supplied Character, never a new default
template. Omitted features remain unchanged. This is a scripted editing adapter,
not an automatic photo-to-likeness optimizer.

```json
{
  "version": 1,
  "grooms": {
    "Beard": {
      "asset": "/MetaHumanCharacter/Optional/Grooms/Bindings/Beards/WI_Beard_S_Stubble.WI_Beard_S_Stubble",
      "parameters": {"Melanin": 0.95, "Redness": 0.0}
    }
  }
}
```

| Key | Default / range | Meaning |
|---|---|---|
| `version` | Required: `1` | Profile schema |
| `sources` | `[]` | Optional verified artifact references: `path`, SHA256 and `storage: external` for absolute external files |
| `grooms` | `{}` | Slot mapping; supported slots: `Hair`, `Eyebrows`, `Beard`, `Mustache`, `Eyelashes`, `Peachfuzz` |
| `grooms.SLOT.asset` | Required for each selected slot | Existing compatible wardrobe asset under `/Game` or `/MetaHumanCharacter`; replaces the slot selection, not an appended duplicate |
| `grooms.SLOT.parameters` | Omitted parameters unchanged | Optional `Melanin`, `Redness`, `Roughness`, each `[0,1]`; these are native material controls, not measured hair colours |
| `eyes.global_tint_linear` | Unchanged | Three linear RGB multipliers in `[0,1]`, applied to both native irises |
| `eyes.global_saturation` | Unchanged; `[0,4]` when supplied | Both irises' native saturation control |
| `eyelashes.type` | Unchanged | `none`, `short_sparse`, `short_fine`, `short_thin`, `long_slight_curl`, `long_curl`, `long_thick_curl` |
| `eyelashes.melanin`, `redness`, `roughness`, `salt_and_pepper`, `lightness` | Unchanged; each `[0,1]` | Explicit native lash material/pigment controls; not measured strand colour |
| `eyelashes.dye_color_linear`, `enable_grooms` | Unchanged | Linear RGB dye `[0,1]` and boolean groom enablement; contradictory disabled/selected lashes are rejected |
| `assembly.pipeline`, `assembly.quality` | `optimized`, `medium` | `optimized` or `cinematic`; optimized quality `high`, `medium`, `low`. Cinematic ignores quality and records that distinction |
| `face_basecolor` | Unchanged | Existing Texture2D in verified native UV space; enables UE 5.8 texture/material overrides |
| `sculpt.reference_landmarks` | Required with sculpt | Exact native control snapshot for this Character; stale geometry is rejected |
| `sculpt.deltas` | Required with sculpt | Unique `{index, delta_cm: [x,y,z]}` edits in native coordinates |
| `sculpt.max_displacement_cm` | Required with sculpt; `(0,2]` | Maximum individual edit and observed whole-face displacement budget |

Sculpt inputs need anatomically justified targets. The adapter does not invent a
mapping between GNM/MediaPipe points and native sculpt controls. Iris multipliers
are not calibrated colour matching, and a GNM atlas is not a native-UV texture.
Explicit lash/pipeline policies are unit-tested; they have not yet been applied
and visually verified on a new personal-head candidate.

### Photo-native feature comparison

Feature extraction is owned by [imageprocessing](../imageprocessing/README.md#reusable-feature-evidence).
This command reads its reference/render manifests, reports projected regional
differences and optionally writes a native iris-profile **proposal**:

```powershell
python -m ailias facefitting compare-features subject --reference C:\features\photo\manifest.json --reference-face-id SOURCE_FACE_ID --candidate C:\features\render\manifest.json --current-tint 1 1 1
```

| Option / output | Meaning |
|---|---|
| `--reference`, `--reference-face-id` | Completed generic feature manifest and exact source face |
| `--candidate` | Single-face `inspect-image --role render` manifest; the role is caller-declared, not authentication of Unreal origin |
| `--current-tint R G B` | Optional actual native tint used for that render; omit for geometry-only diagnostics |
| `geometry_2d` | Signed differences of outer-eye-span-normalized projected ratios; no calibrated-mm or isolated 3D error claim |
| `iris-proposal.json` | Bounded current-tint x reference/render colour ratio, with hash-pinned evidence; directly accepted by `unreal-features --profile` but **not automatically applied** |
| Missing iris/geometry | Explicit review/unmeasurable status, never a zero error or invented colour |

Reports live under `facefitting\feature-comparisons`. Reference/render expression
and lighting may differ. The result always keeps `likeness_verified=False`;
a proposal requires a subsequent native render. This is diagnostic plumbing, not
the missing anatomical optimizer or a completed coordinated appearance pass.

### Native control mapping

Map an existing read-only native export without launching Unreal or modifying
any mesh. All paths are explicit; installed metadata is referenced, not bundled:

```powershell
python -m ailias facefitting native-control-map subject --topology C:\exports\topology.json --inspection C:\exports\report.json --configuration C:\UE\landmarks_config.json --anatomy C:\UE\face_landmarks.json --source-package C:\Project\Content\Head.uasset
```

`--topology` supplies native centimetre vertices; `--inspection` supplies the
exported API controls and source hash. `--source-package` must still match that
hash and the receipt must attest no source save/change. `--configuration` and
`--anatomy` supply the installed canonical vertices and anatomical curves/points.

The mapper tests each control against **canonical native configuration vertices**
within `1e-5` cm. It does not assume configuration sorting equals API ordering.
Coincident exported vertex aliases are not substituted for canonical IDs.
Multiple possible canonical vertices abstain; absent anatomy remains unknown.
This is not source-mesh nearest-surface wrapping or a new geometry fit.

Maps are retained under `facefitting\native-control-maps`. The real retained
inspection mapped **79/79 controls**, but only **20/79** have exact memberships
in the supplied anatomical curves/points. That does not establish a complete
MediaPipe-to-native correspondence or authorize automatic sculpt deltas.

### Native assembly verification

With `unreal-features --assemble`, edits and runtime assembly share one editor session, followed
by an independent editor reload. Validated edits are saved before assembly, so
an assembly failure does not discard that checkpoint. The current runtime
inspector requires bound groom components: baked-only stubble at optimized
LODs is not yet supported by that verification gate, even when native wardrobe
selection persisted. Such runs remain failed, not appearance-approved.

## Current working baseline

There is no active personal Blender baseline. On November 18, 2026, the owner
requested deletion of `hairfree-matched-camera.blend` from diagnostic revision
`20261117T0525Z-hairfree-camera`, together with the active
`data\arka\facefitting\working-baseline.json` selector. Both have been removed.
Original photos, other historical revision inputs and saved native Characters
were not deleted. New full runs must start from retained sampled photos and
create new downstream assets, not restore the retired scene.

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

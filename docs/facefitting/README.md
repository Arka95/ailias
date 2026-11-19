[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Photo-to-character commands

Fit one complete GNM head to multiple photographs, keeping identity separate
from each view's camera and expression. Generate Blender geometry/appearance
previews and hand a hair-free mesh to Unreal.

Scripted feature profiles persist native groom, iris, bounded sculpt and
texture-override edits. These operations do not provide an automatic anatomical
likeness optimizer or calibrated photographic skin transfer.

## Inputs and outputs

| Operation | Required input | Retained output / next consumer |
|---|---|---|
| `video-evidence` | Local MP4/MOV folder, prepared landmarker | Default 1 FPS images capped at 1920 pixels, PTS, expression/pose evidence and PCM audio; explicit all-frame mode is separate |
| `video-create` | Retained sampled-frame manifest, installed models, Blender and UE project | Fresh observations/fit, textured Blender scene and hair-free mesh handoff, native-UV colour, native grooms/iris, independent reload and new editor portrait; likeness remains unapproved |
| `video-fit` | Video evidence, source-matched native responses, native ray correspondences and preserved appearance profile | Experimental shared native identity/per-frame expression fit and bounded **unapplied** sculpt profile; no Blender |
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
python -m ailias facefitting video-evidence subject C:\recordings
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

### Recorded video

Install optional `.[video]` support in the existing environment. Default
`config\video-evidence.json` selects **one frame per second** for routine geometry
work (`frame_mode=sampled`, `sample_fps=1`). Set `frame_mode=all` explicitly for
full-rate expression/lip-sync evidence.
Saved processing images use at most 1920 pixels on the long side, preserve
aspect ratio and never upscale. Detection uses at
most 1280 pixels on the long side. Landmark coordinates and `source_size` use the
saved image grid; `original_source_size` records the input dimensions. Original
recordings and existing exports are unchanged. Images, timestamps/expressions and source-aligned audio live
under `data\ALIAS\facefitting\video\REVISION`; original recordings are unchanged.
Native numerical proposals live under `video-fits`, separately from saved UE
`features` revisions. Strict-neutral and expression-aware selections are distinct.
The latter requires a per-frame expression model; it does not label smiles as
neutral. Audio/lip motion is performance evidence, not neutral nose/jaw geometry.
See [video identity research, tradeoffs and commands](video-identity-investigation.md).

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
appearance profile. Before Unreal import, the command creates a fresh
Blender clay scene, bakes photo colour onto the fitted GNM UVs, omits scalp hair
by default, saves `blender\textured-head.blend` and
exports the hair-free mesh that Unreal actually consumes. The Blender export
must preserve fitted vertices under the declared axis/centimetre conversion.
Missing scenes, requested hair, render outputs or failed Blender processes stop
the run; the command does not fall back to the direct Python OBJ.
No intermediate approval prompt or manual tuning is part of execution.

`--review-config` selects `config\character-review.json` for native-preview
lighting, independently of `config\reanimation.json`. The review profile uses
-3.5 EV; the speaking-video profile uses -5.5 EV.

An appearance policy may set `native_hair_asset` to an explicit native
Wardrobe Item. This overrides the approximate length-based style selector,
but retains current-photo pigment estimates. Keep `hair_strands: 0` when
Blender should remain hair-free; native groom selection is independent.
This is a stock-style approximation, not reconstructed individual strands.
The same appearance policy can include `assembly`, using the native feature
profile's `pipeline` and `quality` settings. Select
`{"pipeline": "cinematic"}` for cinematic assembly or use the
optimized/medium defaults. This does not change Blender
hair or the requested video resolution.
`video-create --assemble` also builds and independently reloads the runtime
Blueprint. Supplying `--voice WAV --ffmpeg EXE` implies assembly and continues
through placement, audio lipsync, direct movie rendering, encoding and full
video/audio verification:

```powershell
python -m ailias facefitting video-create subject --evidence C:\data\sampled-photos\manifest.json --editor $ue --project $project --voice C:\audio\narration.wav --ffmpeg C:\tools\ffmpeg.exe --reel-config config\reanimation.json
```

Native cloud rig/texture requests require explicit `--cloud-services`; the
pipeline fails rather than enabling them implicitly. Without `--assemble` or
`--voice`, the command retains its preview-only behavior. Appearance editing and
assembly share one native revision rather than cloning the Character twice.
Narrated-video requests skip the separate neutral preview and produce verified
video without PNG exports. Whole-alias history pruning is explicit maintenance;
each job still releases its own scratch.
The native feature receipt separates their phase timings.
`--reel-config` controls narrated-video composition independently of
`--review-config`. Existing `unreal-features --assemble` and `reanimation run`
commands remain available for independent operations.

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

The broad search is explicit opt-in, not part of an appearance edit or render.
See [search contracts](dev.readme.md#fitting-restrictions-and-search).

The photo-colour route connects native skin UV export, fresh photo-colour baking and
Texture2D import to native skin, iris, eyebrow, scalp-hair, facial-hair and
eyelash edits. `--appearance-config` uses `config\appearance.json` for the
CUDA device, atlas resolution, photo cap and view count. Stock groom selection,
pigment mapping and landmark-region facial-hair darkness are **approximations**;
the parser has no distinct beard/moustache class. Eyelashes use an explicitly
unmeasured preset. Observed colour retains photo lighting; unseen atlas texels
are labelled median-skin fill, not recovered detail.

For Unreal's own complexion instead of a projected photo atlas, set
`native_appearance` in the appearance policy to native `skin`, `freckles` and/or
`accents` properties, with an explicit `native_hair_asset`. This route skips
native photo-atlas baking/import and explicitly removes the face base-colour
override. Native picker/texture settings do not change geometry or certify a
photo-colour match.
Blender appearance remains independent. Speaking renders must load
`config\reanimation.json`; the brighter `character-review.json` is only a
preview choice, not a replacement video preset.

`fresh_video_completed` means unattended editable-Character
creation plus a **new editor-native preview**, not an assembled animation rig
or accepted likeness. Any failed stage exits **2**. Results and stage receipts are under
`data\ALIAS\facefitting\fresh-video\REVISION\manifest.json`.
This path needs the existing facefitting, imageprocessing and explicitly
selected CUDA runtime prerequisites; it does not install or replace GPU stacks.
See [workflow internals](dev.readme.md#execution).

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
| `hair_strands` | `0` | Blender scalp hair is disabled: no cap, strands or Alembic groom; positive values opt into approximate hair. Unreal grooms are independent |
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
| `eyes.primary_color_u`, `primary_color_v`, `secondary_color_u`, `secondary_color_v` | Unchanged; each `[0,1]` | Both irises' native colour-picker coordinates; not RGB values |
| `eyes.color_blend`, `color_blend_softness` | Unchanged; each `[0,1]` | Native primary/secondary iris colour blending |
| `skin.u`, `skin.v` | Unchanged; each `[0,1]` | Native skin-tone picker coordinates; not measured melanin or photo RGB |
| `skin.roughness` | Unchanged; `[0.85,1.15]` | Native skin roughness multiplier |
| `skin.face_texture_index` | Unchanged; integer `[0,152]` | Installed TS-1.3-F model's 153 native complexion textures; not a geometry or groom edit |
| `freckles.mask` | Unchanged | `none`, `type1`, `type2`, `type3` |
| `freckles.density`, `strength`, `saturation`, `tone_shift` | Unchanged; each `[0,1]` | Native freckle density, opacity and colour; disabling the overlay does not remove marks baked into the complexion texture |
| `accents.REGION.redness`, `saturation`, `lightness` | Unchanged; each `[0,1]` | Symmetric native complexion/lip accents. Regions: `scalp`, `forehead`, `nose`, `under_eye`, `cheeks`, `lips`, `chin`, `ears` |
| `eyelashes.type` | Unchanged | `none`, `short_sparse`, `short_fine`, `short_thin`, `long_slight_curl`, `long_curl`, `long_thick_curl` |
| `eyelashes.melanin`, `redness`, `roughness`, `salt_and_pepper`, `lightness` | Unchanged; each `[0,1]` | Explicit native lash material/pigment controls; not measured strand colour |
| `eyelashes.dye_color_linear`, `enable_grooms` | Unchanged | Linear RGB dye `[0,1]` and boolean groom enablement; contradictory disabled/selected lashes are rejected |
| `assembly.pipeline`, `assembly.quality` | `optimized`, `medium` | `optimized` or `cinematic`; optimized quality `high`, `medium`, `low`. Cinematic ignores quality and records that distinction |
| `face_basecolor` | Unchanged when omitted | Existing native-UV Texture2D enables an override; explicit `null` removes only the face base-colour override and restores native complexion |
| `sculpt.reference_landmarks` | Required with sculpt | Exact native control snapshot for this Character; stale geometry is rejected |
| `sculpt.deltas` | Required with sculpt | Unique `{index, delta_cm: [x,y,z]}` edits in native coordinates |
| `sculpt.max_displacement_cm` | Required with sculpt; `(0,2]` | Maximum individual edit and observed whole-face displacement budget |

Sculpt inputs need anatomically justified targets. The adapter does not invent a
mapping between GNM/MediaPipe points and native sculpt controls. Iris multipliers
are not calibrated colour matching, and a GNM atlas is not a native-UV texture.
Native skin, iris palette and lash fields are checked after an independent editor
reload. Picker coordinates are explicit starting choices, not automatic
photo-colour matching. `skin` does not change texture indices or request a new
sculpt. A skin/freckle/accent/base-colour profile also selects UE's **Skin** preview mode;
Topology or Clay can hide correctly saved skin values. Independent reload
verification checks that Skin mode is enabled. Appearance-only edits must
preserve the sampled native face controls.

Leave `--assemble` off to validate editable Character save/reload first. This
avoids coupling an appearance edit to runtime rig/ControlRig assembly; successful
property persistence is still separate from visual correctness.

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

Maps are retained under `facefitting\native-control-maps`. A canonical vertex
association does not establish anatomical membership or a complete
MediaPipe-to-native correspondence. Missing memberships remain unknown and do
not authorize automatic sculpt deltas.

### Native assembly verification

With `unreal-features --assemble`, edits and runtime assembly share one editor session, followed
by an independent editor reload. Validated edits are saved before assembly, so
an assembly failure does not discard that checkpoint. The current runtime
inspector requires bound groom components: baked-only stubble at optimized
LODs is not yet supported by that verification gate, even when native wardrobe
selection persisted. Such runs remain failed, not appearance-approved.

## Optional baseline

When supplied, a baseline pins source geometry, camera and scene dependencies
for compatible native refinement. Every reference must remain intact.
`video-create` instead starts from the selected retained images and creates
new downstream assets; it does not require or restore a previous baseline.

## Photo appearance preview

| Neutral clay | Photo-colour and approximate hair |
|---|---|
| ![Fitted geometry](../images/pipeline/clay.png) | ![Blender appearance preview](../images/pipeline/appearance.png) |

Generic eyes, texture seams, residual photographed lighting and inferred hidden
anatomy remain. The preview renderer also scales inherited light energies to
8%; its displayed skin tone is not a calibrated reference.

## Runtime handoff

Assemble a verified Character with `unreal-features --assemble`, then pass its
compiled Blueprint to [reanimation](../reanimation/README.md) with narration.
This does not require another head fit. Native package/rig/media checks do not
certify photographic resemblance.

## Native coordinate and hair boundaries

The conform adapter uses a matched perspective tracking view and native
`(GNM X, Z, Y) * 100` target coordinates. Static import instead uses
`(GNM Z, -X, Y) * 100`; these conventions must not be mixed.
Nominal scaling is not physical calibration.

An exported Alembic alone is not a validated bound groom. Check strand identity,
axis conventions, root alignment, binding, deformation and LOD behavior.
Adding a beard or copying a texture file alone does not establish a working
deformable appearance. See [handoff internals and tests](dev.readme.md).

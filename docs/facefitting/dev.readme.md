[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Facefitting internals

## Package ownership

The CLI dispatches commands; workflows compose the implementation packages.
Unit tests live in responsibility-specific `tests` subpackages.

| Package / module | Owns |
|---|---|
| `evidence` | Retained-image loading, per-image analysis, semantic extraction and explicit video sampling |
| `geometry` | Head models, camera transforms, correspondence, optimization and fit/OBJ serialization |
| `appearance` | UV rasterization, photo texture projection, observed-feature comparison and native profile mapping |
| `unreal` | Native asset operations, editor-process request/receipt handling, persistence and portrait capture |
| `blender` | Scene/render/export process adapters and workers; required by the fresh full pipeline |
| `workflows` | End-to-end stage ordering, source ownership, failure reports and final result assembly |
| `configuration.py` | Validated fitting, frame-quality/video and appearance policies |
| `artifacts.py` | Saved fit/appearance manifest inspection and dependency verification |

Geometry code does not import workflow orchestration. Shared OBJ/fit I/O lives
in `geometry\artifact_io.py`; per-image analysis lives in
`evidence\frame_analysis.py`, so processing retained photos does not import the
video decoder module. Native feature validation remains the same pure-Python
contract used by the standalone Unreal feature worker; mapping does not launch
an editor or duplicate that schema.

Internal Python paths and callable names changed, not the public commands or
serialized contracts. `video-create --evidence`, existing stage/status values,
worker request fields and the `fresh-video` artifact directory remain stable.
Historical generated worker copies and personal assets are not rewritten.
`unreal\character_portrait.py` renders an assembled runtime Blueprint;
`unreal\character_portrait_worker.py` captures an editable editor Character.
Neither filename implies likeness approval.

## Verified Blender-to-Unreal run

On November 18, the public `video-create` command completed unattended as
revision `17b9ff17ee584c24b6d8f5d42e981ec5`, after the ownership refactor and
required Blender handoff were implemented. It reused 82 retained images,
fitted 81 views and did not decode video or reuse previous personal predictions,
fits, scenes, native responses or appearance profiles.

| Recorded stage | Seconds |
|---|---:|
| UE preflight, retained-photo/model loading | 13.126 |
| Fresh photo observations and parsing | 35.859 |
| Shared identity fitting | 14.312 |
| Fresh Blender photo texture, scenes, hair and export | 52.656 |
| Native Character factory/conform/save | 95.266 |
| Independent native surface reload and UV export | 78.218 |
| Native-UV photo texture bake | 9.047 |
| Native Texture2D import | 25.187 |
| Evidence-to-native profile mapping | 0.016 |
| Native appearance apply and independent reload | 88.546 |
| New editor-native portrait | 52.391 |
| Workflow wall time, including artifact finalization | **466.375** |

These are shared-host observations, not an isolated throughput benchmark or a
claimed speedup over earlier runs. The command was not paused between stages.
The Blender stage created 14,000 approximate strands, including 984
photo-supported and 582 inferred rear root vertices. Its hair-free geometry
export was the actual input to Unreal. All **157 unique checked artifact
references** match; independent native surface max/RMS reload displacement is
exactly **0 cm**. The resulting native actor exposes scalp hair, brows, beard,
moustache and eyelash groom assets. The retired Blender scene/selector remain
absent.

Outputs are under `data\arka\facefitting\fresh-video\17b9ff17ee584c24b6d8f5d42e981ec5`:
`blender\textured-head.blend`, its `appearance-0.png` and `native-preview.png`.
The saved Character is
`/Game/Ailias/Features/F5ed881af1db14fc785dbf2993b77fdf0/Character.Character`.
Both actual new images were opened. The Blender hair cap has a visibly hard
edge, the native draft is dark-lit, photo colour retains illumination, and
native grooms remain stock approximations. This verifies execution and
persistence, not accepted likeness or an assembled animation rig.

Delivery checks passed 253 unit tests against an isolated snapshot of the
staged source, plus compilation and public CLI argument checks. The broader
working-tree cohort passed 265 tests before the real integration run.

## Fresh sampled-photo path

`workflows\photo_to_character.py:run_photo_to_character_pipeline` and `unreal\character_creation_worker.py` implement
`video-create`. `evidence\photo_processing.py` reads only image references and capture
metadata from retained samples; old landmark, expression and semantic results
are discarded. Two CPU workers prefetch at most four decoded images per batch.
Fresh dense inference and face parsing are serialized per model instance;
owned models close on success or failure. Per-photo feature and region evidence
is retained under `photo-evidence`, including explicit unresolved beard,
moustache and eyelash classes rather than fabricated segmentation.
This path fits the generic GNM prior from the freshly computed observations
using held-out selection across 48, 96 and all 170 supported head-identity modes.
The run manifest retains the model's actual correspondence count, full final
per-view residual/support statistics and template displacement quantiles.
These distinguish a changed mesh from independently measured surface detail.
`blender\appearance_scene.py` then projects
current photo evidence onto the genuine fitted GNM UVs, creates a new clay
scene through `blender\clay_rendering.py`, and runs the existing appearance
worker on that same-run scene. The worker saves a textured `.blend`, four
views and a hair-free export. Blender scalp hair is now disabled by default
(`hair_strands: 0`): no cap, curves or Alembic groom is created. Positive budgets
retain the optional approximate-hair path. Photographed beard/brow pigment in
the texture is not erased, and Unreal's native groom mapping is unchanged.
Unreal imports this Blender
export, not a bypass OBJ. Geometry/basis preservation and requested hair are
mandatory stage results. No historical scene or forged approval is used.
The earlier manually reviewed `appearance` command retains its approval gate.
The pipeline then calls
the existing head-only native conform worker with **no source Character**.
The native factory creates a new asset in a UUID namespace; a second editor
process compares exported surface topology and vertices with a post-commit,
post-save reference; sculpt-control drift is retained separately.
The conform worker also compares the live solve to each commit and the save,
so changing the reference cannot conceal geometry lost during commit.
Max/RMS Euclidean differences are recorded in centimetres; the existing
0.0001 cm tolerance is not loosened. Previously saved personal assets
are never used as shape/appearance initialization. Native geometry export
requires Slate: use `-ExecutePythonScript -RenderOffscreen -d3d12`, not a
`-run=pythonscript -NullRHI` commandlet.

The run records stage/wall times and every failed boundary. Native target-pose
finalization through `commit_posed_state_as_a_pose` now passes a real independent
surface reload with exactly zero displacement; finalization must precede the
face/body commits. It also moves the head into canonical body coordinates.
The photo baker explicitly inverts a least-squares rigid pre/post-finalization
alignment and records its nonrigid residual, rather than treating body-height
vertices as the original fitted camera coordinate frame.
Real native UV export, fresh colour baking/import, native appearance save/reload
and actual groom-component instantiation now pass, and a complete run produced
a new image. Actual image review caught a vertical UV reversal and too-wide
framing, despite successful property checks. The native atlas raster adapter
now converts top-left-origin Unreal V to the shared bottom-left-origin OBJ
rasterizer convention. The preview uses native face-control bounds, not
skeletal component bounds that include shoulders. Both fixes have focused units
and pass the delivered full fresh run. SceneCapture's FOV is horizontal:
the requested physical vertical span is multiplied by output width/height
before deriving that angle. Earlier direct-native delivery run
`b58c0c440faa4df5900e613ad8faa7a1`
completed in 540.265 s, and its actual new close-up was opened. All 131 checked
input/output references match and native surface max/RMS reload drift is zero.
That run did not include Blender and predates the ownership refactor; it does
not validate the revised full path or establish visual likeness acceptance.
The run cannot pass by applying the previous
manual appearance profile. The older `workflows\video_character_refinement.py`
is a source-Character refinement orchestrator, not this from-scratch path.
Focused fresh-worker units mock UE; real GPU/native execution is opt-in through
the command. Existing fit/conform units cover the reused components.

`unreal\worker_process.py` owns native request/receipt handling. A nonzero process exit
still retains a readable worker failure receipt in the parent report. Missing
or malformed receipts, timeouts, worker-declared failures and source-integrity
failures cannot become successful stages. Photo failures include the face ID
and preserve their original exception chain; cleanup errors are recorded
without replacing the primary failure.

`unreal\skin_texture_worker.py` exports per-corner native skin UVs, retaining seams,
within the verifying editor process: the geometry-export mesh is not a saved
cross-process asset. Its other operation imports a fresh Texture2D.
`appearance\skin_texture_projection.py` projects the same run's
photos/cameras into that atlas; approximate nearest-prior expression offsets
do not alter the native rest mesh. Both fitted-GNM and native-skin texture
passes share `appearance\photo_texture_baking.py` for current camera/photo
selection, bounded temporary arrays, visibility blending and explicit atlas fill.
They use their respective genuine UVs; the Blender atlas is not silently
substituted for native skin UVs. Owned temporary photo arrays are removed on
success and failure. Coverage and unseen-fill limitations remain explicit.
`appearance\profile_mapping.py` consumes current regional colours and facial-hair
darkness estimates, validates the resulting native profile and records all
stock-groom/iris/lash assumptions. `apply_character_features(use_working_baseline=False)`
duplicates only this run's new Character, without loading an earlier personal
baseline. `unreal\character_portrait_worker.py` spawns that saved editor-native Character,
assembles its preview grooms and captures a new 540x960 draft after nonblocking
editor ticks, using lighting/exposure from the existing reanimation policy.
It does not claim an assembled runtime or request cloud services.

## Video identity path

`evidence\video_sampling.py` owns PyAV decode, processing-grid JPEG/PTS evidence, frame-wise
MediaPipe expression predictions and clock-aligned PCM audio. It writes only a
fresh exclusive UUID revision, so decoding does not hold the global session
writer lock or modify shared session state. Failures retain a failed manifest;
frame-count/PTS/budget/audio-gap failures cannot claim a complete timeline.
The numerical fitter still holds the session lock. As of November 18, routine
geometry extraction defaults to one frame per second; explicit `frame_mode=all`
retains the original expression timeline. Processing JPEGs are capped at 1920
pixels on the long side without upscaling or aspect changes; evidence coordinates
use that same grid and retain `original_source_size` separately. The detector
input limit remains 1280 pixels on the long side. Neither originals nor existing
full-rate/full-resolution exports are changed.

`geometry\video_identity_fitting.py` fits native response identity modes jointly with per-frame
camera and similarity-aligned GNM expression modes. It does not call Blender or
edit UE. Category labels are fixed from initial views during held-out scoring;
candidate camera changes cannot dodge category nonregression checks. Predicted
mesh displacement, flips/collapse and stretch are checked before producing a
source-specific profile. Its linear mesh prediction is not an actual saved
Character export. See [research and acceptance](video-identity-investigation.md).

On November 17 at 23:38 +05:30 the owner reopened geometry work using the new
recordings. This supersedes the earlier geometry freeze for this new native
workflow only; Blender remains stopped and approved appearance is preserved.

## Fitting restrictions and search

The November 18 audit compared actual neutral vertices, stored Blender inputs,
candidate scores and final camera fits, rather than assuming that a high mesh
vertex count implied dense observations. Run `c4841ba98cdf4a838a6ddb7683cf9ff0`
completed the public pipeline in 410.735 seconds with zero Blender hair objects,
zero strands and no Alembic groom. Testing 170 identity modes alone still chose
48; that did not establish improved likeness.

| Restriction or finding | Implementation / consequence |
|---|---|
| 17,821 mesh vertices vs 468 mapped landmarks | `HeadModel.audit` now records correspondence support, directly constrained triangle vertices and available identity capacity; neither mesh density nor displacement is a measurement count |
| Actual mesh propagation | Prior run moved 17,764 vertices by over 1 mm in uncalibrated model units; Blender's stored fitted geometry matches the fit within floating-point export precision |
| Binding focal floor | Some final focal factors reached 0.450005; shared CPU/CUDA bounds now allow 0.1..5 and camera initialization includes wider-angle hypotheses |
| Incomplete expression capacity | The former common 100-mode validator incorrectly prevented the supported 150-mode lower-face region; regional capacities are now 100/100/150 |
| Original photographs absent | `evidence\reviewed_photos.py` accounts for the complete scan inventory and prepares every accepted portrait across groups from original pixels, without old fit/dense predictions |
| Original-photo camera coordinates | Review crops localize detection only; crop landmarks are mapped back to full, EXIF/ICC-normalized source coordinates before fitting and texture projection |
| Camera solver degeneracy | An SQPNP numerical failure now tries EPNP for that focal hypothesis instead of aborting the whole run; complete initialization failure remains an explicit view rejection |
| Multiple people in an accepted crop | The reviewed face box selects among freshly predicted faces by overlap; ambiguous identity is reported, not substituted with another person |
| Identity/expression ambiguity | `geometry\identity_search.py` searches their prior strengths independently, along with capacities and expression bounds; raw held-out point RMS and category breakdowns are retained |
| GPU memory vs evidence | `geometry\cuda_optimization.py` accumulates shared Schur terms over bounded Jacobian batches without subsampling views |
| Visual accountability | `evidence\fit_comparison.py` saves every selected fit's observed/projected landmark overlay; green is detected, magenta is fitted |

`--fit-search-config` uses the caller's explicit Cartesian grid. Every trial is
fresh within the same evidence run, while generic camera/expression controls
are shared only within matching configurations. All candidate receipts and
parameter arrays are retained in `fit-search`; the selected identity is refitted
on the full prepared cohort before Blender and native creation. A failed
generic control does not suppress personalized trials; those start from the
original initialized cameras instead and retain the control error.
Failed candidates and validation scope remain explicit. Lowest 2D landmark
error is not a global-best 3D shape guarantee: landmarks remain predictions,
unseen anatomy remains prior-driven and silhouette/photometric surface fitting
is not implemented by this search.

The corrected public run `252acd173452429388a7e40718bce9a6` completed all
15 stages in **4309.235 seconds**. Its inventory accounts for all 192 original
source photographs: 188 curation-accepted portrait records were added to 82
retained video samples. Fresh processing yielded 179 original-photo views and
81 video views; ten inputs did not yield an unambiguous usable detection.
There were 211 training and 49 identity-held-out views, then a final refit on
all 260. Old predictions, personal fits and video decoding were not reused.

All **162 trials completed without failure**. Candidate 112 selected 96 shared
identity modes, 100/100/150 expression modes, identity/expression penalties
`1e-6`/`5e-7` and expression bounds of +/-6. Its category-balanced held-out
weighted RMS was **0.0303005 outer-eye spans**, versus **0.0386592** for the
matching generic-head control (about 21.6% lower on the same metric).
This is not a comparison with the earlier video-only metric or independent
3D ground truth. Final optimization converged in 12 iterations.

The final focal-factor range was **0.147872..3.101750**, away from the new
0.1/5 limits. No identity coefficient reached +/-3 (maximum absolute 2.3974).
However, **346 per-view expression coefficients remain within 0.001 of +/-6**,
and the winner uses the weakest tested priors and largest expression bound.
The search therefore does not establish an unconstrained optimum; more
expression freedom could also absorb identity errors. The model still lacks
dense photometric/silhouette constraints and independently measured unseen
anatomy. All 260 views carry non-neutral warnings; 141 carry blink warnings.
They remain expression-aware inputs, not certified neutral captures.

Template-relative median/p95/maximum displacement was **5.415/11.849/14.774 mm
in uncalibrated model units**; 17,799 vertices moved by over 1 mm.
Blender preserved the fitted geometry and reported zero scalp-hair objects,
zero strands and no exported groom. Native surface max/RMS reload drift was
**0 cm**, and all **768 unique referenced artifacts** matched their hashes.
These checks establish transfer/persistence, not likeness.

Actual new clay comparison, textured Blender front view, native preview and
front/profile overlays from both input sources were opened and inspected.
The shape differs visibly from its generic prior, but forehead/neck texture
boundaries remain, the native preview is dark/red-lit, and native grooms are
stock approximations. The Blender scalp is hair-free; photographed texture
pigment is not equivalent to a generated hair object. Likeness remains
unapproved. Outputs are under
`data\arka\facefitting\fresh-video\252acd173452429388a7e40718bce9a6`;
the new Character is
`/Game/Ailias/Features/Fba66c8950d994a84b2bb7163a80e8e52/Character.Character`.
An isolated snapshot of the exact staged source passed **390 functional unit
tests**, compilation and public CLI option checks. Native/GPU integration
evidence comes from the actual full run, not those mocked unit tests.

## Fitting architecture

| Component | Responsibility |
|---|---|
| `geometry\head_model.py` | Pinned GNM assets, validated numeric loading and surface correspondence |
| `evidence\landmark_detection.py` | Compatibility imports from `imageprocessing.dense`; no duplicate detector implementation |
| `geometry\optimization.py` | Camera initialization, region weights, CPU reference and shared evaluation |
| `geometry\cuda_optimization.py` | Batched deformation/projection/loss, automatic derivatives and Schur/LM optimizer |
| `geometry\artifact_io.py` | Fit parameter serialization and OBJ export retaining per-corner UV seams |
| `workflows\head_fitting.py` | Fresh jobs, held-out candidate comparison, all-view refit, exports and integrity |
| `blender\clay_render_worker.py` | Neutral-clay review scene/rendering; not the numerical fitter |

One shared identity explains all views; camera/expression parameters vary per
photo. The neutral export omits per-photo expression. The 468 mapped face points
constrain the exterior; iris observations are retained without invented surface anchors.
Hidden anatomy remains prior-supported, not measured.

The CUDA optimizer uses forward-mode Jacobians and a Schur complement to
eliminate independent per-view blocks before updating identity. Adaptive damping
rejects objective regressions. CPU least squares uses the same objective with
different numerical optimization; neither backend is a likeness certificate.

## Appearance and native boundaries

| Component | Responsibility / invariant |
|---|---|
| `workflows\appearance_generation.py` | Matching geometry-review gate, new revision, atlas/export assembly and cleanup |
| `evidence\semantic_parsing_worker.py` | Fresh OpenCV CUDA semantics isolated from Torch imports |
| `appearance\uv_rasterization.py`, `appearance\cuda_texture_projection.py` | UV rasterization, expression-aware projection, coverage/depth/semantic tests and linear-light blending |
| `blender\appearance_scene_worker.py` | Photo material, generic eyes, approximate scalp hair, scene and hair-free export |
| `unreal\static_mesh_import.py` | Static import, package hashes, declared basis and bounds checks |
| `blender\tracking_view_worker.py` | In-memory frontal tracking camera without rewriting the baseline scene |
| `unreal\head_conforming.py`, `unreal\head_conforming_worker.py` | Local native Character solve, state commit and saved-package verification |
| `geometry\photo_camera.py` | Optional original-photo contour view using retained fitted extrinsics; full-frame alpha compositing avoids a cropped-camera principal-point error |
| `unreal\character_features.py`, `unreal\character_features_worker.py` | Immutable-source Character duplication, validated sculpt/iris/texture/groom edits, optional native assembly and independent reload |
| `unreal\character_portrait.py` | Hash-verified runtime review and one-frame neutral rendering through the reanimation worker; no narration or fitting |
| `appearance\feature_comparison.py`, `appearance\iris_color_matching.py` | Compare generic per-photo/render evidence and write bounded, source-pinned iris edit proposals without launching an editor |
| `geometry\sculpt_control_mapping.py` | Match exported API controls to installed canonical native vertex IDs; only exact named anatomy memberships, no nearest-region guesses |
| `unreal\sculpt_response.py`, `unreal\sculpt_response_worker.py` | Source-pinned finite-difference sculpt-to-mesh calibration on an isolated unrigged copy; restore the entire native state and verify its exported mesh |
| `geometry\surface_correspondence.py` | Associate rendered landmarks with visible triangles of that same native mesh; exact vertex-ID rebinding abstains on changed triangulation |

Native appearance profiles also expose the installed UE skin picker `u`/`v`,
skin roughness, iris primary/secondary picker coordinates and colour blending.
The adapter commits `skin_settings.skin` through `commit_skin_settings`, and
verifies those values independently after reopening the saved Character.
Ranges follow installed `MetaHumanCharacterSkin.h` and
`MetaHumanCharacterEyes.h`; skin roughness is a multiplier in `[0.85,1.15]`,
not the normalized `[0,1]` used for groom roughness.

The November 17 20:31 +05:30 owner direction freezes geometry and stops all
Blender work. Current work prioritizes an appearance-only Character checkpoint
without runtime assembly, followed by native editor review. Native picker
coordinates are provisional appearance choices, not calibrated photo albedo.

Appearance revision `f866335bf8cc4436836dd53036411918` applied native skin
`u=0.4`, `v=0.65`, roughness `1.06`, iris palette/tint/saturation, thick dark
native brows and short dark lashes to a duplicate of geometry candidate
`7f4db166ad1e4516b7212d674cca5550`. Editable Character saving and a **fresh
editor reload passed**, without assembly/cloud requests or the earlier ARKit
save exception. All sampled native face controls had **zero displacement**;
the source package and protected dependencies remained unchanged. The 22
focused native-feature units passed.

The apply worker recorded 2217.625s wall time, including 2133.367s in save;
the independent reload took 39.578s. Native skin commit itself took 0.654s.
The save log contains a long timestamp gap, audio-device timeouts and a later
texture wait; this does not isolate the cause or establish active CPU cost.
No save-speedup is claimed. Native visual review and runtime assembly remain
separate gates; this is not calibrated photo-colour or likeness acceptance.

The subsequent live viewport inspection found that revision `f866...` retained
`MetaHumanCharacterSkinPreviewMaterial.DEFAULT`, whose installed display name is
**Topology**, not Skin. Both texture and material overrides were disabled; the
display mode, rather than a texture override, hid the colour. Property-only
checks had missed this. Skin/base-colour edits now select `EDITABLE` (**Skin**)
before registration, and reload verification rejects topology/clay mode.
The regression selection passes 23 focused native-feature units.

Corrected Skin-mode revision `4a2af06bd4e44a7ba1b24eac58078046` passed an
independent reload and was actually inspected with textured skin in the UE
Character editor. The following reference-pinned appearance passes also saved
and independently reloaded without runtime assembly or cloud requests:

| Revision | Native grooming / actual viewport observation |
|---|---|
| `c9476d2ce5e545fab648ec181c4c0323` | Short Messy hair, Thick brows, Short Stubble beard/moustache. Visible, but hair too short/upright and moustache too faint for reference `59d8a173...`. |
| `bce3643f69eb4c82b0bddc4476201234` | Medium Side Swept Fringe, Flat Thick brows, Short Uneven beard and Short Full moustache. Actual hair reached the neck and was rejected as too long; beard also too heavy. |
| `8e5fd408aab64af884668817a34ad12e` | Short Side Swept Fringe, Flat Thick brows, Medium Stubble and Short Full moustache. Saves/reloads in 43.234s/34.797s. Owner subsequently approved beard/moustache at 22:10 +05:30; preserve them rather than making further unsolicited facial-hair changes. |
| `922cffc50b024f3e8674e8ad0c50105f` | Skin-only material pass: roughness `1.06 -> 1.12`, complexion index `0 -> 7`, no freckle overlay, and regional forehead/cheek/nose/chin/lip accents. Actual UE viewport inspected; fewer prominent marks and softer shine. |
| `f6b61b60c2ec4189bb1e87b8a47bf66f` | Refined tone `u=0.35,v=0.9` using the displayed native picker and observed photo chromaticity; lip lightness `0.46` softens the initially too-dark `0.42` adjustment. Saved/reloaded in 128.594s/44.188s and inspected in the actual UE viewport. |

These are real UE viewport observations, not just selected asset names. The
latest draft preserves the source and protected hashes and reports zero sampled
face-control displacement. See the [actual photo/UE comparison](images/ue-photo-appearance-comparison.png)
and [provenance](images/ue-photo-appearance-provenance.json). No likeness approval,
de-lit photo skin, new runtime rig or full-mesh geometry validation is implied.
Preview overlays were hidden only for presentation; that did not resave the
Character package.

The installed synthesis model's `ts_params.json` declares 153 characters.
Its `texture_attributes.json` labels texture 0 as low wrinkles/no stubble/medium
marks and texture 7 as low wrinkles/no stubble/low marks. This supports choosing
a cleaner native complexion without blurring away all detail or repainting the
approved groom. The adapter exposes the native integer texture index, freckle
mask/parameters and symmetric regional accents; omitted fields stay unchanged.
Native texture-index changes are rig-restricted in the editor: this trial uses
the existing **unrigged** Character, not an implicit unrig/rebuild workflow.

The worker now records the pre-edit appearance as well as requested/reloaded
values. All 25 focused native-feature units pass, including source isolation,
complexion/lip persistence, invalid ranges, and Skin-mode gating for accent-only
or freckle-only profiles. No configured linter exists in `pyproject.toml`.
Latest-to-approved checks confirm identical groom assets/material parameters,
eyes and lashes, zero sampled face-control movement, unchanged approved package,
and valid protected dependency hashes. The roughness multiplier is the native
stored value; at `1.12`, the normalized editor tool displayed `0.9`.
See [skin before/after](images/ue-skin-before-after.png). These are visually tuned
native materials, not measured/de-lit albedo or a new identity certificate.

Official control semantics: [Epic Skin Material Tools](https://dev.epicgames.com/documentation/metahuman/skin-material-tools?lang=en-US).

Photo projection uses source cameras, portrait offsets and actual resize ratios.
Face parser boxes already use portrait coordinates; subtracting the source offset
again is incorrect. Coverage excludes fill and seam padding.
Depth-splat visibility is approximate, not exact triangle rasterization.

| Appearance issue | Current behavior |
|---|---|
| Skin colour | Observed photo RGB, not de-lit albedo; first selected skin median anchors bounded 0.85-1.15 gains |
| Preview darkness | Inherited lights are multiplied by 0.08; material is lit again despite photographed shadows |
| Hair | Procedural scalp approximation; photographed moustache/stubble colour is not a bound beard groom |
| Native transfer | Conform sends vertices, curves and camera, not Blender texture/UV/groom appearance |
| Native rig textures | Existing runtime draft uses native source textures, not a bake of the GNM atlas |

The darkness contributions have not been isolated by an A/B render.
Do not silently treat an RGB atlas as calibrated skin reflectance.

The local comparison `dea1edf777ae4745baca47cc84abe24f` consumes the frontal-photo
and old rejected-render feature revisions documented in the
[upstream validation record](../imageprocessing/dev.readme.md#feature-extraction-and-native-diagnostics).
It emits projected ratio differences and an **unapplied** iris proposal. The
reference is smiling and the native frame nearly neutral; these differences do
not isolate native shape errors. That comparison did not generate a native render.
The combined focused units passed 56 tests; new lash/pipeline settings have only
unit-level validation at that milestone.

The subsequent photo-contour trial `500a0046ca8f47d0b2b45d519451cb7e` tracked
16 curves from the glasses-free reference `59d8a173...-00`, without a new GNM fit
or Blender render. Its duplicate incorrectly retained the inherited full rig.
Feature job `c40c186d2cd9431987c628efe0469786` assembled 223 assets with bound
native brows/lashes and dark iris settings, but failed final Character saving
with an ARKit ControlRig/skeleton binding error. Diagnostic review
`800e902f2fbd45ed8d5e00d275ef5059` is a genuinely new frame, not a passed
save/reload result; facial geometry remains visibly wrong. No beard/moustache
was selected, in accordance with the face-first priority.

The worker now removes an inherited full rig on the isolated conform copy,
and saves/checks the built Character before instantiating the runtime actor.
The relationship between inherited rig state and the save failure is not
established as its sole cause. Corrected conform
`9bf935cd4d0943c78b09073ed6a86cc8` saved successfully with
`inherited_face_rig_removed=True`; this is not a new rendered likeness result.
The save-order correction still needs a real assembly/reload validation.
The focused photo-camera/conform/control-map/native-feature units passed
**48 tests**. Another assembly is deferred while source geometry is diagnosed.

The retained source fit already has an identity-quality warning:
held-out frontal error changes only from `0.02141002` for the generic prior to
`0.02121983` for the selected identity, approximately **0.9% improvement**.
The reported 10.8% overall improvement was view-category-balanced and largely
came from oblique views; it is not frontal likeness acceptance.
For reference `59d8a173...-00`, the photo nose-width/outer-eye-span ratio is
`0.37224`, versus `0.48173` for the neutral source projected through the retained
camera, about **29.4% wider**. The expression-deformed fitted projection is
`0.44584`, still about **19.8% wider**. Inner-eye-span ratios are `0.36646`
in the photo versus `0.41854` in that neutral projection. These are conditional
2D diagnostics using existing predicted landmarks and expression-fitted cameras,
not isolated 3D anatomy measurements or permission to apply those percentages
directly as native sculpt deltas. Inputs: the retained fit's `manifest.json`,
`cameras.json`, `observations.json`, `fit.npz` and pinned GNM correspondence.
All selected ratio anchors have positive retained fitting weights and head-mask
support. Across 34 frontal views, the median expression-deformed nose ratio is
about 5.9% wider than each photo estimate: the selected photo's larger discrepancy
must not be treated as a universal correction.

Native mapping revision `017304af8c8148c09f2e0978d0053482` used the retained
`native-control-topology-20261117T1105Z` inspection and unchanged source package.
All 79 API controls resolve to one canonical configuration vertex each; 20 have
exact anatomical memberships. Four controls have coincident exported aliases:
choosing a single nearest mesh vertex would miss their canonical definitions.
The map uses configuration-vertex coincidence instead, with unknown/ambiguous
associations abstaining. It does not move any controls.

Seven new synthetic mapping units cover ordering/aliases, absent and ambiguous
matches, invalid coordinates/IDs, mesh scoping and changed package dependencies.
The combined mapping/native-policy/comparison selection passed **32 tests**.
No new UE process, Blender fit, scene copy or native render was needed.

### Direct native geometry calibration

The original-photo tracker returned only 16 curves in the retained conform:
lips, philtrum, eyelids and nasolabial folds. There were **no nose, jaw, forehead
or brow curves** in that receipt. Repeating this solve is not an implemented
whole-face photo optimizer.

The internal `native_response.measure_response` helper runs under the caller's
`store.lock()`, duplicates an **unrigged** Character into a fresh namespace,
and measures X/Y/Z perturbations of requested sculpt controls. Omitting
`indices` selects all controls. It does not change the source or apply a fitted
identity. The response manifest pins the source, selected baseline and worker;
`reference.f32` contains `vertex_count x 3` little-endian float32 centimetres.
`responses.f32` contains one equally sized derivative block per receipt entry,
ordered by control then axis, in output centimetres per requested centimetre.
`triangles.json` records that measurement's native topology.

Two engine integration details were exercised locally:

- Native export reads **committed** Character state. Before `commit_face_state`,
  the preview landmarks changed while the exported mesh remained unchanged.
- The export API synchronizes Content Browser and asserted in commandlet mode.
  The worker therefore uses offscreen editor `-ExecutePythonScript`, not
  `-run=pythonscript`, and explicitly quits its owned editor.

The 1,397-value native coefficient vector is retained only as an **opaque state
snapshot** for restoration. Its entries are not assumed to be independent shape
modes, and the fitter must not guess semantic indices.

Pilot revision `62847393c8134323aff105eaaab4cc78` measured all three axes of
known mouth-corner control 32 against 33,845 exported vertices. A requested
0.05cm X shift produced a maximum mesh displacement of approximately 0.00942cm:
the native sculpt response is not a one-to-one translation. The final exported
mesh and sampled control positions restored exactly; source and all protected
baseline hashes passed. The pilot editor lifecycle took 88.36s; its individual
response measurements took approximately 0.69-2.41s. These are calibration
timings, not an end-to-end likeness speedup.

Full revision `8f3934755cf742afbb7c4fb0ecd1587d` subsequently completed all
**237 directions across 79 controls** in one editor lifecycle of 663.063s.
Canonical topology remained unchanged across its probes, the restored entire
mesh had zero displacement, and source/baseline hashes passed. This is measured
deformation data, **not an applied or accepted identity fit**.

For photo correspondence, the retained old neutral-render camera projects six
explicit native eye/mouth corners within approximately 0.19-2.74 pixels of the
cached detector predictions. Visible ray/triangle association supports all
468 non-iris predictions on that native reference mesh. Rebinding to the pilot
mesh by **exact canonical face vertex IDs**, not triangle-list order, retains
467 and abstains on landmark 384 after native retriangulation. This is not
cross-topology nearest-surface wrapping. Exact numerical reprojection validates
the camera/ray implementation, **not anatomical accuracy**; detector semantics
and photo pose/expression remain uncertain.

## Coordinate and persistence contracts

| Boundary | Mapping / check |
|---|---|
| OBJ export | `(GNM Z, X, Y) * 100` |
| Imported static preview | `(GNM Z, -X, Y) * 100` |
| Native Character target/camera | `(GNM X, Z, Y) * 100` |
| Baseline | Verify every pinned source; no new fit or saved `.blend` copy during native refinement |
| Conform | Commit face/body state, save package, check actual file |
| Rig/assembly | Saved packages, compiled actor/rigs and independent reload/render are separate from API return values |
| Native features | D3D12/offscreen worker; original package and every baseline dependency must retain their hashes |
| Native wardrobe | Edit `get_preview_collection`, call `on_edit_preview_collection` to persist selections/parameters, then independently reload |

Absolute scale remains uncalibrated. A large raw displacement may be translation;
use translation-removed/aligned diagnostics and actual visual comparisons.
RMS zero across rigging only means those sampled native points stayed fixed,
not that Blender-to-native likeness is correct.

UE 5.8's editing preview owns a separate collection. Editing only
`character.internal_collection` after registration does not update the assembled
preview. The adapter uses the preview collection and explicitly propagates its
changes back before saving. Groom slots use `set_single_slot_selection` to avoid
duplicate selections. Every requested exposed parameter is checked after a fresh
editor process; geometry must remain unchanged for appearance-only jobs.

`--assemble` reuses a full rig and high-resolution textures when present.
Editing and building run in one editor process; a second process independently
reloads the Character and assembled actor. Feature properties, sculpt displacement
and original-source hashes remain mandatory checks. Edits are saved before
building, and feature state is checked again after assembly. Missing cloud
prerequisites fail unless explicitly enabled. Runtime verification
instantiates the compiled actor, checks its Face/head/RigLogic and every requested
groom's binding, then destroys only that test actor. Neutral review uses a saved
one-frame sequence without an audio performance; ordinary reanimation retains
its existing audio solve/export path.

The bound-component gate currently rejects baked-only short grooms; it does not
verify native stubble baked into skin at lower LODs. A retained failed assembly
can be inspected with `unreal-review --diagnostic`, without promoting its source
status or claiming visual acceptance.

Each feature receipt records `worker_seconds` and non-overlapping `stage_seconds`
for loading, edit registration, preview assembly, feature checks, runtime
assembly, Blueprint compilation, package saving and actor inspection. Host
`apply_runtime`/`verify_runtime` durations also include editor startup and shutdown;
worker stages are subsets, not additional wall time. Native phases are measured
on the next execution; removing a process is not itself a measured speedup.

The local full-rig Character requires D3D12 even for edit registration; NullRHI
asserted in native skinned-mesh code. Do not silently substitute it. Job-owned
temporary files use shared finalization; source/native packages are retained as
hash-referenced evidence, not blindly deleted by filesystem cleanup.

## Configuration

See [fitting/appearance key tables](README.md#configuration) and
[native-tool policy](../generation/README.md#configuration).
Model manifests retain checksum, revision/source and license records.
Workers are copied by content hash into each job so application edits do not
silently rewrite the implementation that produced existing results.

## Evidence

Retained benchmark records under `data\arka\facefitting\benchmarks` separate
end-to-end wall time from optimizer-only measurements.

| Check | Retained result / interpretation |
|---|---|
| Fresh CPU vs CUDA pipeline | 372.859s vs 42.187s on matching source/split evidence; 8.838x measured wall-time ratio |
| Balanced held-out error | CPU 0.02508408 vs CUDA 0.02470406 of eye span; conditional projected error, not scan accuracy |
| Mesh integrity | Finite vertices, noncollapsed triangles and source hashes; no self-intersection proof |
| Native handoff/runtime | Static import, Character conform, full rig and saved runtime actor/render exist |
| Visual result | Personal draft still fails resemblance and photo-appearance transfer |

`full-pipeline-cpu-cuda.json` records the benchmark comparison;
`reviews\personal-runtime-character.json` pins native package/rig/assembly/reel
evidence. The [root comparison](../../README.md#current-state---below-expectations)
shows the unresolved quality gap.

## Validation and development

```powershell
python -m unittest discover -s src\ailias\facefitting -t src -v
```

The committed package reorganization passed **246 units** across facefitting
and the affected application, resource, imageprocessing and reanimation suites.
Those checks used an isolated copy of the exact staged source, excluding
pre-existing experimental edits. This naming/ownership change did not rerun
GPU/UE reconstruction; the real execution results above predate the refactor.

Units use synthetic geometry/fake observations and editors. Explicit hardware
tests are separate:

```powershell
$env:AILIAS_CUDA_FITTING_TESTS = "1"
python -m unittest discover -s src\integration_tests\facefitting -p test_cuda.py -v
python -m unittest discover -s src\integration_tests\facefitting -p test_appearance_cuda.py -v
```

Do not infer native appearance, groom binding or likeness from unit success.
See the [workflow investigation](../generation/metahuman-workflow-investigation.md)
and [hair guidance](hair-fitting-guidelines.md) for detailed alternatives/evidence.

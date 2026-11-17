[Back to REPO_ROOT](../../README.md) | [Usage](README.md)

# Head fitting internals

| Module | Responsibility |
|---|---|
| `settings.py`, `config\facefitting.json` | Explicit device, mode counts, fitting budgets and regularization |
| `model.py`, `config\facefitting-models.json` | Pinned public downloads, retained license provenance, safe numeric NPZ loading, surface correspondence checks |
| `observations.py` | CPU MediaPipe 478-point predictions, expression scores, foreground checks and reversible source coordinates |
| `fitting.py` | Camera initialization, normal-based visibility heuristic, region balance, held-out split, explicit CPU reference and shared evaluation |
| `cuda.py` | Batched CUDA deformation/projection/loss, forward-mode derivatives and block-sparse Levenberg-Marquardt with a Schur complement |
| `pipeline.py` | Fresh jobs, immutable dependencies, candidate selection, all-view refit, OBJ/overlays and render invocation |
| `blender_worker.py` | Prior/fitted neutral-clay views and independent review scene, not fabricated SfM output |
| `appearance.py`, `config\appearance.json` | Geometry-approval gate, bounded new appearance jobs, texture assembly, export/integrity checks and temporary photo cleanup |
| `texture.py`, `texture_cuda.py` | Per-corner UV rasterization/view ranking; CUDA projection, linear-light blending, depth-splat/semantic coverage and hair evidence |
| `semantics_worker.py` | Fresh OpenCV CUDA segmentation in a process separate from Torch; portrait-grid face boxes, not full-source boxes |
| `appearance_worker.py` | UV/photo material, transparent generic eye shells, approximate hair cap/strands, new scene/renders and separate head export |
| `unreal_handoff.py` | Real hair-free head import, native package hashes and imported coordinate/bounds checks; not MetaHuman creation |
| `unreal_conform.py`, `unreal_conform_worker.py`, `tracking_worker.py` | New local Character fits from verified imports, matched perspective tracking, native +Y target/camera conversion, state commit and package verification; no rig/cloud |

## Research decisions in plain language

| Question | Decision and reason | Limitation / alternative |
|---|---|---|
| Why not continue COLMAP? | The cleansed 64-photo set produced only three verified match pairs; a known head shape supplies coherent anatomy when photo triangulation fails | A prior can look plausible without being accurate. Native reconstruction remains separately available |
| What is GNM? | A learned statistical model: template plus weighted identity/expression shape changes. Use pinned Google GNM 3.0 head assets, whose official repository licenses supplied weights and code Apache-2.0 | It is not a text-to-image model. Unseen anatomy is inferred through model correlations, not measured |
| Why not the previously investigated 3DDFA/Basel assets? | Do not assume academic-use-only model terms cover this use; choose the permissively licensed GNM prior instead | Revisit another model only with verified terms and measured benefit |
| Where do image features come from? | MediaPipe predicts 478 points and expression scores. Use the first 468 for the available surface correspondence; retain the ten iris points as observations, not invented GNM anchors | Predicted relative Z is not metric depth. No per-point visibility/confidence is supplied by this adapter |
| Is the community correspondence trustworthy? | Verify numeric weights, indices, triangle membership and active skin-exterior contributions locally. Reject invalid rows by geometry, not landmark IDs | Two active rows fail exterior support and are excluded. Other mapping errors remain possible; author accuracy claims are not independent validation |
| Why separate identity, pose and expression? | One identity must explain many photos while cameras and expressions vary independently. Otherwise a smile or camera tilt can become permanent anatomy | The decomposition is approximate; expression bases can absorb identity differences |
| Why fit in source coordinates? | Analysis alignment is reversible; transform points back to the EXIF-oriented source before perspective fitting | A centered crop's center is not necessarily the camera principal point. Focal length and absolute scale remain uncalibrated |
| What counts as visible? | Require foreground alpha, exterior support and a front-facing prior normal; balance constraints across facial regions | This is not a full z-buffer, glasses detector or proof of visibility. Occluded points can still be predicted incorrectly |
| How is the winner chosen? | Hold out every fifth face-ID-sorted observation within each initial dense view category. Compare category-balanced mean errors, require at least 1% improvement and at most 5% frontal regression | The held-out cameras/expressions are still fitted. This is a conditional 2D check, not independent 3D scan accuracy |
| Why not the overall median? | Many frontal views can hide meaningful oblique improvements. Keep the median as a diagnostic, but balance view categories for selection | The initial trial failed the median-only criterion (0.95% gain); balanced evaluation showed 9.42% gain with 0.72% frontal regression. Historical failed report is preserved |
| What makes it slow? | Initial SciPy code repeatedly evaluated the full batch to estimate derivatives numerically. About 313 of 326 seconds were numerical fits/validation | Merely queuing more CPU work does not accelerate this calculation. First remove repeated derivative work; do not weaken accuracy gates |
| Where should CUDA be used? | Keep fit data, deformation, projection, robust loss and automatic differentiation on one explicitly selected GPU; process every eligible view as one batch | This is a different optimizer over the same objective. Compare quality as well as runtime. I/O, tiny camera initialization and Windows MediaPipe remain CPU work |

Zero joint rotations use the GNM bind-pose linear shape terms; this module does
not implement articulation/skinning or claim an expression rig. The neutral
export deliberately omits per-photo expression coefficients.
FP32 CUDA computation is explicit; scalar diagnostics and final products are
transferred back to CPU. GPU failure is reported rather than silently selecting
CPU or another device. No GPU percentage or speedup is inferred from tensor size.

The CUDA optimizer uses smooth bounded parameters and automatic differentiation,
whereas the CPU reference uses sparse finite-difference least squares. Both use
the same Huber-like landmark objective, identity/expression priors, positive-depth
penalty and publication checks. There is no background queue service, parallel
candidate execution or cross-run inference replay cache yet.

`jacfwd` and `vmap` compute the per-view Jacobian blocks together. A Schur
complement eliminates independent camera/expression blocks before solving the
shared identity update; batched CUDA Cholesky solves the local blocks. This avoids
finite-differencing the full batch for every parameter. Adaptive damping rejects
steps that worsen the same objective.

The shared environment already contained TorchVision 0.28.0+cu126, which requires
Torch 2.13.0. The installed/pinned Torch 2.13.0+cu126 preserves that pairing;
system CUDA 13/OpenCV was not replaced. Same-process OpenCV CUDA GEMM and
TorchVision CUDA NMS both passed. The initially attempted 2.9.1+cu130 install was
superseded, not retained as a second environment.

## Evidence

Authorized CPU baseline `20261116T145947Z-57cdcc65`:
64 curated inputs, 63 usable dense faces, 53 training / 10 held-out identity
views. Candidate 96 wins over candidate 48. Category-balanced held-out error
falls from 0.02769366 to 0.02508408 of outer-eye-corner span (9.423% reduction).
Frontal mean rises from 0.02120392 to 0.02135557 (0.715% regression).
The final all-view refit converges in ten evaluations; its median error is
0.02245693 of eye span, **not a held-out score**.

The generated head has finite vertices and no collapsed triangles; area-ratio
1st/99th percentiles are 0.8315/1.2630. Self-intersections are not checked.
All retained artifacts and dependencies passed hashes. Overlay and comparison
images were inspected: the shape is coherent, but resemblance is unapproved,
glasses/expressions remain uncertain, and the original nonfrontal camera roll
was wrong. The renderer now constructs an explicit Y-up frame; the old scene
and exact original worker bytes remain preserved.

First failed trial wall time was 326.453s, before rendering.
Accepted CPU run wall time was 372.859s, including 24.078s in the Blender CUDA
stage. Numerical-fit wall time totals 334.515s, about 90% of the accepted run.
These are real command/stage measurements, not estimated GPU speedups.

An initial GPU L-BFGS experiment was rejected: the 53-view generic stage took
20.234s instead of 91.047s but worsened the objective by about 30%.
The accepted Schur/LM benchmark on frozen observations took **8.390s** (10.85x
faster), with objective 0.000973765 versus CPU 0.000975214.
Its retained report is `data\arka\facefitting\benchmarks\cuda-schur-a15979ce.json`.

Fresh CUDA job `20261116T153125Z-44390de9` repeated dense inference, candidate
comparison, all-view fitting and rendering. Outer CLI wall time was **42.187s**
versus **372.859s** CPU: `372.859 / 42.187 = 8.838x`, an 88.686% reduction.
Persisted pipeline-only time was 40.922s; do not compare that with CPU outer time.
The 96-mode candidate's category-balanced held-out error fell from 0.02770004
to 0.02470406 (10.816% improvement); frontal error also improved.
Final all-view median was 0.02210726, not a held-out score. Its final refit took
0.203s; peak process CUDA allocation was 413,934,592 bytes, not total VRAM use.
Both jobs passed artifact/model/source integrity checks. The new comparison
renders were visually inspected: front/45/90-degree views are now upright and
the mesh is coherent, but clay rendering does not establish likeness.
Input references/hashes, eligible/excluded face IDs and the entire training/
held-out split match exactly. The retained comparison record is
`data\arka\facefitting\benchmarks\full-pipeline-cpu-cuda.json`.

See the [researched MetaHuman route and remaining gates](../generation/metahuman-workflow-investigation.md).

## Appearance evidence and limits

The owner approved only clay geometry. Revision `20261116T163340Z-14973ca7`
used 16 selected photos, fresh verified OpenCV CUDA graph execution, Torch CUDA
UV projection and Blender Cycles GPU rendering. Its persisted total was
49.297 seconds, including source/model integrity checks and four 768-pixel views;
this is not a benchmark against the earlier fitting jobs.
It preserves the approved scene/OBJ hashes; Blender's float32 copy differs by at
most `1.49e-8` model units. Its new eye material reveals the existing iris/pupil
geometry through the previously opaque outer shell, not new measured eyes.

Photo projection uses each view's expression-deformed head and source camera,
subtracts the full-source portrait offset, then applies the actual resize ratio.
Semantic parsing instead consumes `entry.box_xywh`, which is already in the
portrait grid: subtracting the source offset twice was found and regression-tested.
Visibility is front-facing normal plus dense UV-sample minimum-depth splatting,
**not** exact triangle rasterization. Require near-opaque supported labels;
exclude glasses, background, scalp hair and the mouth interior from skin colour.
The neck below the chin is deliberately unmeasured/neutral, avoiding face-colour
spill from unconstrained neck projections. RGB gains are bounded, blending uses
linear light, and eight-texel seam padding does not increase reported coverage.

Hair votes also require foreground semantics and the depth test. Smoothed
photo-supported scalp roots are extended into explicitly inferred unobserved
scalp; deterministic procedural strands approximate volume, not individual hairs.
Generic eye optics, roughness, unobserved skin and hidden anatomy remain priors.
Do not treat UV coverage or a successful render as likeness/albedo approval.

### Hair-free camera diagnostic (November 17, 2026)

One new Blender-only proposal is retained under
`data\arka\facefitting\diagnostics\20261117T0525Z-hairfree-camera`.
`comparison.png` places an unchanged-proportion original-photo crop beside an
actual new Cycles render. The earlier appearance worker uses fixed orthographic
review angles, not the fitted source perspective camera; those views cannot
isolate eye/nose proportion differences against a photograph.

The diagnostic uses the existing estimated perspective camera for face
`84fbcc146e51fd1b296eaa8d5d2a87634e7cf2000ff1b880fe75b30e6c1dc1c1-01`
(`PXL_20210908_081939952.jpg`). The Blender conversion agrees with the existing
projection equation within 0.000242 render pixel, which verifies conversion,
**not** camera calibration or likeness. It retains the approved GNM geometry,
photo texture, generic eyes and neutral expression. Only the two approximate
scalp-groom objects are hidden reversibly in a new scene.

The actual render remains visibly imperfect: temple/forehead texture boundaries,
generic eyes, eyelid/expression mismatch and facial-hair differences remain.
Hiding scalp hair does not repair these or recover unseen skin. The reference
was excluded from texture assembly but was used in the final all-view identity
fit, so this is not independent held-out validation. No new shape deformation,
photo warping, hair synthesis or Unreal work ran. CUDA Cycles on the RTX 4070 SUPER
took 2.972 seconds to render; the complete Blender process took 11.709 seconds.
The source scene/geometry/texture/reference hashes were checked before and after;
existing revisions remain unchanged. `runtime.json`, `process-runtime.json`,
`request.json`, `diagnostic-report.json` and `blender.log` retain the evidence.
See [hair fitting guidelines](hair-fitting-guidelines.md) for the separate
research on capture occlusion versus final identity.

UE import `99504db87d434eceb922b52cf96b02f7` saved eight native packages,
including static head and photo texture, in 16.438 seconds. Bounds verify the
declared upright mapping `UE=(GNM Z,-X,Y)*100`; absolute scale is uncalibrated.
The imported candidate excludes the separate hair cap/strands. Initial trials
found and fixed Windows OBJ texture links, Unreal Array JSON serialization,
valid hyphenated package names and the importer's axis convention. Export
materials must resolve to the adjacent retained texture; only actual saved
packages count as a successful static handoff.

Native head-only job `73508d43f43540eaafa792c527abc083` subsequently saved
`/Game/Ailias/ConformedHeads/C73508d43f43540eaafa792c527abc083/HeadOnlyDraft`.
Native execution took 81.375s; this is not an end-to-end benchmark.
It tracked 16 facial curves, committed face and body states, and a separate
editor process reloaded the Character and extracted its actual editing mesh.
`data\arka\unreal\native-review.json` and `conformed-head-front.png` retain that
post-reload evidence. The first +X-facing attempt was distorted despite the
solver returning true. Rotating both supplied vertices and camera to native
`(GNM X,Z,Y)*100` corrected the gross distortion: final face-landmark bounds
are `[-8.896,-5.301,14.565]` to `[8.673,17.481,40.627]`. Landmark displacement
includes placement and is not a likeness score. Eye/eyelid detail, surface
refinement and final likeness remain unapproved.

Factory creation/edit registration and the local solve succeeded without optional
Core Data; engine warnings disable skin editing rather than all Character work.
No rig/cloud request, production skin bake, assembly or animation validation ran.
The separate Alembic imported and reloaded as a real Groom asset, with 28,000
curves and 2,800 guides, not the intended 14,000 curves. Native inherited struct
fields require `get_editor_property`; direct Python attributes failed in the
initial receipt step after the package had already saved. Count discrepancy,
axis/scale/root alignment and final skeletal-head binding remain unresolved.

## Validation and development

```powershell
python -m unittest ailias.facefitting.tests.test_fitting ailias.facefitting.tests.test_observations -v
python -m unittest ailias.facefitting.tests.test_texture ailias.facefitting.tests.test_appearance -v
python -m unittest ailias.facefitting.tests.test_unreal_conform ailias.facefitting.tests.test_unreal_handoff ailias.generation.tests.test_unreal -v
$env:AILIAS_CUDA_FITTING_TESTS = "1"
python -m unittest discover -s src\integration_tests\facefitting -p test_cuda.py -v
python -m unittest discover -s src\integration_tests\facefitting -p test_appearance_cuda.py -v
```

Units use synthetic geometry/fake observations and never import Torch or probe
hardware. The separate explicit CUDA tests compare projection with the CPU
reference, check gradients and recover known identity/opposing expressions.
Actual full-photo trials remain separate from units. A likeness sign-off,
silhouette/surface calibration and rig/animation validation remain open.
The fitting milestone passed **162 isolated tests** and two explicit CUDA tests.
The appearance implementation passed **169 isolated tests**, followed by
**21 focused appearance/Unreal regression tests** after final handoff changes.
The new opt-in CUDA sampler test passed on a known-colour synthetic surface.
The native Character addition passed **20 focused conform/import tests**, including
missing packages, malformed receipts, pixel cleanup and no-cloud failure paths.
Actual photographic/Blender runs and the saved UE assets above were verified
separately. Local Character conform/reload and unbound Groom import were run
as described above; final likeness, rig/binding/animation and Mermaid rendering
were not validated.

## Primary references

- [Official GNM source and weight license](https://github.com/google/GNM/tree/dfd3e4b96bda2f0061c6013a458b435cf3cb519f)
- [Official GNM NumPy implementation](https://github.com/google/GNM/blob/dfd3e4b96bda2f0061c6013a458b435cf3cb519f/gnm/shape/gnm_xnp.py)
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [Unofficial correspondence and its stated limitations](https://gist.github.com/pundirmr/ecf0c0bb5c93fc1a9534083a8b6d255e)
- [SciPy least squares and numerical Jacobians](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
- [Official PyTorch wheel index used for the matching runtime](https://download.pytorch.org/whl/cu126)
- [PyTorch forward-mode automatic differentiation](https://docs.pytorch.org/docs/stable/generated/torch.func.jacfwd.html)

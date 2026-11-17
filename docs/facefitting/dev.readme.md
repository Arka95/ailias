[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Facefitting internals

## Fitting architecture

| Component | Responsibility |
|---|---|
| `model.py` | Pinned GNM assets, validated numeric loading and surface correspondence |
| `observations.py` | MediaPipe predictions, expression evidence and source-coordinate restoration |
| `fitting.py` | Camera initialization, region weights, CPU reference and shared evaluation |
| `cuda.py` | Batched deformation/projection/loss, automatic derivatives and Schur/LM optimizer |
| `pipeline.py` | Fresh jobs, held-out candidate comparison, all-view refit, exports and integrity |
| `blender_worker.py` | Neutral-clay review scene/rendering; not the numerical fitter |

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
| `appearance.py` | Matching geometry-review gate, new revision, atlas/export assembly and cleanup |
| `semantics_worker.py` | Fresh OpenCV CUDA semantics isolated from Torch imports |
| `texture.py`, `texture_cuda.py` | UV rasterization, expression-aware projection, coverage/depth/semantic tests and linear-light blending |
| `appearance_worker.py` | Photo material, generic eyes, approximate scalp hair, scene and hair-free export |
| `unreal_handoff.py` | Static import, package hashes, declared basis and bounds checks |
| `tracking_worker.py` | In-memory frontal tracking camera without rewriting the baseline scene |
| `unreal_conform.py`, `unreal_conform_worker.py` | Local native Character solve, state commit and saved-package verification |
| `native_features.py`, `native_features_worker.py` | Immutable-source Character duplication, validated sculpt/iris/texture/groom edits, optional native assembly and independent reload |
| `native_review.py` | Hash-verified runtime review and one-frame neutral rendering through the reanimation worker; no narration or fitting |

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
python -m unittest ailias.facefitting.tests.test_fitting ailias.facefitting.tests.test_observations -v
python -m unittest ailias.facefitting.tests.test_texture ailias.facefitting.tests.test_appearance -v
python -m unittest ailias.facefitting.tests.test_unreal_conform ailias.facefitting.tests.test_unreal_handoff -v
python -m unittest ailias.facefitting.tests.test_native_features ailias.facefitting.tests.test_native_review ailias.reanimation.tests.test_reanimation -v
```

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

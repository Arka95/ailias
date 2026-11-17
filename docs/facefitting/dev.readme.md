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

Absolute scale remains uncalibrated. A large raw displacement may be translation;
use translation-removed/aligned diagnostics and actual visual comparisons.
RMS zero across rigging only means those sampled native points stayed fixed,
not that Blender-to-native likeness is correct.

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

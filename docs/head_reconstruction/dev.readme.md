[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Head reconstruction internals

## Responsibility

`head_reconstruction` owns image-to-mesh numerical work and Blender products.
It does not own native Character editing or the end-to-end execution chain.
Public commands route through `facefitting\cli.py`; cross-domain composition
lives in `pipelines`.

| Area | Implementation responsibility |
|---|---|
| `evidence` | Source-coordinate observations, video sampling, reviewed-photo preparation and parsing workers |
| `geometry` | Model loading, camera/identity/expression optimization, search, coordinate conversion and mesh serialization |
| `appearance` | UV rasterization, photo projection, coverage and texture baking |
| `blender` | Clay/appearance scenes, tracking views and coordinate-preserving exports |
| `pipeline.py` | Standalone photo-fit job, review products and artifact checks |
| `settings.py`, `artifacts.py` | Domain policy validation and retained-job loading |
| `reconstruction_data.py` | Typed evidence, numerical results, neutral geometry and stage diagnostics for the feature-based API |

## Fitting

`geometry\head_model.py` loads pinned model assets and correspondence.
`optimization.py` defines camera initialization, weighting, shared evaluation
and the CPU reference. `cuda_optimization.py` performs batched differentiation
and Schur-complement/Levenberg-Marquardt updates: independent per-view blocks
are eliminated before updating shared identity; damping rejects regressions.

`identity_search.py` evaluates configured capacity/prior/bound combinations
with conditional held-out identity checks. Compatible zero-identity reference
solutions are reused only within that search. `identity_fitting.py` performs
the selected all-view fit. `view_batch_size` changes memory use, not evidence.
`artifact_io.py` serializes parameters and preserves per-corner UV seams.

Mapped face observations constrain the exterior. Iris observations do not
receive invented surface anchors. Neutral export omits per-photo expression.
CPU and CUDA share the objective, not an identical numerical update sequence.

### Retained facial-feature boundary

`evidence\feature_observations.py` converts `FaceFeatures` keyed by view ID into
existing numerical observation records. It requires the retained pixel-grid
dimensions and finite dense points/support, derives outer-eye span and suppresses
out-of-image support without mutating predictions. Invalid records retain their
view ID and reason. It performs no image I/O, neural inference or identity matching.

`geometry\feature_reconstruction.py` exposes `reconstruct_head_from_features()`.
Its stages are observation conversion, `prepare_head_views()`,
`fit_shared_head_identity()` and `construct_neutral_head_mesh()`. The caller
supplies the loaded model and explicit `FitPolicy`; no stage runs image
inference, opens source images, exports files or starts a native editor.
The short orchestration function owns status/failure transitions only.

`identity_fitting.py` separates generic-reference fitting, candidate evaluation,
frontal-preserving selection and final all-view parameter restoration/refitting.
Completed candidate diagnostics survive later solver failures. The 1% minimum
held-out improvement and 5% maximum frontal regression are unchanged named
constants near their owner. Per-view expression remains separate from shared identity.

`HeadReconstructionResult`, `HeadFittingResult`, `HeadFitParameters`,
`HeadFitSummary`, `HeadMesh` and `ReconstructionReport` replace the new API's
output dictionaries. Nested inputs, exclusions, candidate scores, runtime,
per-view metrics, mesh quality and failures have scoped record classes.
Orientation and backend-specific scalar metadata remain small explicitly scoped
maps rather than becoming arbitrary root reports. `.to_dict()` and `str(...)`
convert NumPy arrays into strict JSON data explicitly; nonfinite values fail.
No class implements mapping/dictionary emulation.

`geometry\fitting_results.py` adapts existing CPU/CUDA backend records into the
typed fitting result, retaining backend-specific provenance. Backend array
contracts remain internal numerical integration boundaries.
Photo-to-character now calls `fit_shared_head_identity()` and passes its typed
result through overlays, photo projection and Blender scene construction.
`search_shared_head_identity()` returns a typed final fit and supplies typed
candidates to its export callback. The dictionary fitting wrapper and legacy
diagnostic-copy helper were deleted after their caller and tests migrated.
The photo-to-character manifest is version 2: `fit` is a serialized
`HeadFitSummary`; `reconstruction` retains typed input/candidate/mesh diagnostics.
Neutral geometry passes `construct_neutral_head_mesh()` before any Blender/native
handoff, including after capacity search. NPZ array names are unchanged.
The standalone `facefitting fit` CLI and older photo evidence extraction remain
separate pending migration steps, not alternate implementations of the new API.

The new path disables only legacy angle rejection, not camera initialization,
foreground/template support, minimum-view, held-out identity or mesh-quality gates.
`prepare_views(report=...)` diagnostics are translated into typed exclusions even
when the five-view check fails. `HeadReconstructionError.report.failure` retains
the original exception category/explanation, with the original exception as cause.
CPU/CUDA dispatch remains policy-controlled.

`geometry\mesh_quality.py` owns the shared surface-deformation check, also reused
by native-video fitting through its existing dictionary compatibility function.
The feature API returns typed neutral geometry, fitting parameters and reports in memory;
no artifact lifecycle, editor connection, workflow migration or automatic save
is implied. `save_fit()` accepts the typed fitting result and preserves the
existing NPZ layout. Synthetic CPU tests cover actual feature-to-fit execution,
typed JSON serialization, explicit OBJ/NPZ export and per-stage failure diagnostics;
they do not validate real-photo likeness or installed GNM assets.

## Photo appearance

`appearance\pipeline.py` enforces the matching geometry-review dependency.
UV rasterization and CUDA projection apply visibility, depth and semantic
support before linear-light blending. Coverage distinguishes observed texels
from neutral fill. Parser workers isolate incompatible native imports.
`blender\appearance_scene.py` owns scene construction and the hair-free export.

An atlas in GNM UVs cannot be assigned directly to native MetaHuman UVs.
Native photo projection requires a separate native-surface export and
`appearance\skin_texture_projection.py`; native application remains downstream.

## Interfaces and lifecycle

Keep per-photo source coordinates and reversible transforms. Validate hashes
before consuming retained evidence. Shared image decoding and dense detection
come from `imageprocessing`; generic execution/checksum helpers come from `common`.

Numerical functions are independent of editor lifetime. Task-scoped detectors
release their handles; job owners delegate file finalization to
`resource_management`. Native workers run under different Python/import
constraints and are not ordinary in-process host adapters.

## Tests

```powershell
python -m unittest discover -s src\ailias\head_reconstruction -t src -q
```

Synthetic geometry/fake observations cover projection, optimization bounds,
search reuse/invalidation, export coordinates, appearance coverage and failures.
See [cross-module checks](../facefitting/dev.readme.md#validation-and-development)
for pipeline and opt-in CUDA commands.

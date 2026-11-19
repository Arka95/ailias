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

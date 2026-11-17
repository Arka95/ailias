[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Generation internals

## Architecture and contracts

| Component | Responsibility |
|---|---|
| `inputs.py` | Curation/source checks, seed-subject/view selection, reusable hardlinks and masks |
| `matching.py` | Verified view-graph measurements from transient COLMAP SQLite |
| `cameras.py` | World-to-camera poses, crop-adjusted intrinsics and measured reprojection errors |
| `mesh.py` | Preserve per-corner texture coordinates during native mesh conversion |
| `pipeline.py` | Unique jobs, viability gates, stage commits and finalization |
| `tools.py` | Owned executables, scoped DLL paths, timeouts and bounded logs |
| `blender_worker.py` | Minimal coordinate-preserving cleanup, exports and GPU diagnostics |
| `unreal.py`, `unreal_worker.py` | Prerequisite checks, static import and legacy solved-Identity adapter |

Native tools run synchronously through argument/request files and file receipts.
Input identity includes source/portrait hashes, curation, masks/policy and selected
views. Reusing an identical hardlinked view is not cached reconstruction.
The module does not mutate the core manual session state.

## Configuration and gates

The [usage guide](README.md#configuration) owns every effective policy default,
its meaning and PoC differences. `GenerationPolicy` validates counts, numeric
bounds and explicit devices; tool paths resolve relative to the selected config.

| Gate | Publication condition |
|---|---|
| Verified graph | Minimum connected count/fraction before mapping |
| Sparse recovery | Registered count/fraction, sparse point count and measured reprojection error |
| Dense recovery | Minimum fused foreground sample count |
| Mesh/cleanup | Minimum face count and actual retained outputs |
| Static import | Saved native package files and hashes, not a success receipt alone |

Statuses distinguish blocked viability, failed execution, interruption and
completed configured work. `complete` does not mean complete human anatomy.
Partial external-project packages are reported, not automatically deleted.

## Research decisions in plain language

| Decision | Rationale / limit |
|---|---|
| Keep photogrammetry separate from prior fitting | COLMAP triangulates matched image detail; it does not fit semantic face anatomy |
| Mask native inputs | Suppress background/clothing, without inventing facial surface |
| Preserve original projection | Aligned analysis crops are not valid replacements for camera-aware input |
| Keep raw observations/cameras | Meshing can infer unsupported surfaces; polygons alone are weak evidence |
| Use explicit PoC policy | Permit visual fragment experiments without weakening standard gates |
| Preserve scale uncertainty | `SIMPLE_PINHOLE` intrinsics/SfM scale are estimated, not metric calibration |

The coherent shared-identity GNM implementation is in
[facefitting](../facefitting/dev.readme.md), not a missing COLMAP feature.
Detailed research/ablations remain in [selection research](selection-research.md).

## Tests

```powershell
python -m unittest ailias.generation.tests.test_generation ailias.generation.tests.test_unreal -v
```

Fake native/editor objects and synthetic images cover option forwarding, input
identity, missing outputs, partial receipts, package hashing and cleanup.
No model/editor/GPU work runs in these units.

Explicit native tests generate synthetic objects, not personal heads:

```powershell
python src\integration_tests\generation\blender_smoke.py --report C:\diagnostics\blender.json
python src\integration_tests\generation\native_smoke.py --report C:\diagnostics\native-generation.json
```

## Validation

| Area | Existing evidence / limit |
|---|---|
| Tools | Pinned COLMAP 4.2.1 CUDA and Blender 4.5.15 distributions verified locally |
| Synthetic native pipeline | Eight cameras, recovered surface, textured output and three nonflat Blender renders |
| Real-photo strict path | Insufficient connected matching/registration; no complete sparse head |
| Permissive real-photo path | 6,164 textured faces in 13 disconnected fragments; not a usable full head |
| Head-cleansed replay | Three verified pairs among 64 photos; no sparse model |
| Unreal | Static import and separate facefitting-native conform/runtime demonstrated; not proof of this legacy Identity adapter's end-to-end assembly |

Run manifests/ablations preserve exact versions, policies, logs and retained
artifacts. A synthetic pass validates integration, not real facial reconstruction.

## Cleanup

Finalizers discard databases, undistorted images, depth/normal caches and
uncommitted partial exports. Keep committed mesh/atlas/cameras/fused observations,
bounded logs and manifests. Hardlinked canonical views remain reusable.
See [global dependency/ownership safeguards](../dev.readme.md#resource-cleanup).

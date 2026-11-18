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
| `unreal_session.py`, `unreal_session_worker.py` | Project-owned persistent editor, single JSON mailbox, PID/creation-time ownership, OS writer lock and asynchronous completion |
| `blender_worker.py` | Minimal coordinate-preserving cleanup, exports and GPU diagnostics |
| `unreal.py`, `unreal_worker.py` | Prerequisite checks, static import and legacy solved-Identity adapter |

Blender/COLMAP and explicit cold Unreal checks use synchronous subprocesses.
Migrated Unreal adapters call `execute_unreal` with an explicit worker, request
and terminal status, not inferred historical command lines. The persistent
bootstrap uses Slate ticks to admit one hash-pinned worker module per job;
asynchronous MRQ/capture work retains that module until its terminal receipt.
The dispatch callback is unregistered while a synchronous worker is executing,
then restored even after a worker exception. Native MetaHuman builds pump Slate;
reentering Python dispatch there can consume a pending native exception.
Unique job IDs and receipts prevent a stale completion from finishing a new job.
Per-job console variables are restored before admission reopens. Renderer
completion removes its delegate without releasing editor keep-alive; one-shot
workers retain their prior exit behavior.

Only the owned bootstrap may end the persistent editor. Shutdown, dead/recycled
PID detection and project exclusion are separate from alias `Store.lock`.
Shutdown and the `start --wait` launcher retain an OS process handle rather
than reopening a terminating PID, which Windows can reject with access denied.
Native failures and unknown/incomplete receipts fail closed. Pending jobs cannot
be overwritten during the interval before the next engine tick; host timeout
does not imply native cancellation. Per-operation logs contain bounded session
metadata; the owned engine's diagnostic output remains in `Saved\AiliasSession\editor.log`
for its lifetime, replaced on the next owned startup.

Persistent and independent/one-shot migrated launches explicitly select D3D12
**and SM6**. Job admission checks the actual `PCD3D_SM6` startup metadata, not
the GPU's advertised capability. New material-compilation failures reject a
job even if assembly exported assets and wrote a success receipt. One-shot
checks scan the complete log before truncating it; persistent checks stream
only new output in bounded chunks and retain errors until native completion.
This prevents accepting Unreal's default/checkerboard material as baked skin.
The two known standalone `MF_skin_fuzzApply`/`MF_skin_bakedInputs` anonymous
thumbnail previews are excluded, not their actual runtime/master materials.
See the [native failure investigation](../reanimation/pipeline-optimization-investigation.md).

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
python -m unittest ailias.generation.tests.test_unreal_session ailias.generation.tests.test_unreal_session_worker ailias.generation.tests.test_unreal_validation -v
```

Fake native/editor objects and synthetic images cover option forwarding, input
identity, missing outputs, partial receipts, package hashing and cleanup.
No model/editor/GPU work runs in these units.

Explicit native tests generate synthetic objects, not personal heads:

```powershell
python src\integration_tests\generation\blender_smoke.py --report C:\diagnostics\blender.json
python src\integration_tests\generation\native_smoke.py --report C:\diagnostics\native-generation.json
python src\integration_tests\generation\unreal_session_smoke.py --project $project --editor $ue --report C:\diagnostics\session-check\report.json --blueprint /Game/Existing/BP_Character.BP_Character
```

The session check is explicitly opt-in: two native probes, optionally two
180x320 one-frame MRQ renders of a supplied existing Blueprint. It never fits,
rigs or assembles a head. It leaves the working editor available, reports the
actual PID/timings and does not certify skin/likeness. Omitting `--blueprint`
checks only the dispatcher, not asynchronous rendering.

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

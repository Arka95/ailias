[Back to REPO_ROOT](../../README.md) | [Reanimation guide](README.md)

# Persistent Unreal execution and pipeline latency

| Field | Value |
|---|---|
| Owner | ailias repository owner |
| Status | Research and implementation proposal; persistent execution is not implemented |
| Last updated | 2026-11-18 |
| Services in scope | Photo preparation, fitting, native Character editing/assembly, portrait rendering, encoding |
| Reviewers | Repository owner; additional reviewers: *(fill alias)* |

**Finding:** use one owned, persistent Unreal Editor for ordinary operations,
not a new process per stage. Keep an independent saved-asset verification at
the publication boundary, not after every parameter adjustment. Running without
a visible editor window is already implemented; running these MetaHuman editor
operations without an initialized editor engine is a different problem.
Do not refactor or restart the current generation to test this proposal.

## 1. Problem Statement

- **Gap A - process lifecycle.** `NativeTools.run` calls `subprocess.run` for
  every operation. Feature application/build and independent verification are
  separate launches; neutral review and narration rendering launch again.
  The feature worker already combines editing and assembly in one registration.
  Sources: `src\ailias\generation\tools.py:58`,
  `src\ailias\facefitting\unreal\character_features.py:70`,
  `src\ailias\facefitting\unreal\character_portrait.py:79`,
  `src\ailias\reanimation\pipeline.py:82`.
- **Gap B - worker lifetime.** Rendering is asynchronous, but its completion
  callback releases `-ExecutePythonScript` keep-alive. A reusable session must
  complete the job without ending the editor. Some other workers explicitly
  call `quit_editor`; they cannot be blindly imported as persistent jobs.
  Sources: `src\ailias\reanimation\unreal_worker.py:230`,
  `src\ailias\facefitting\unreal\sculpt_response_worker.py:171`.
- **Gap C - larger costs outside startup.** The 162-configuration identity
  search consumed 3523.735 / 4309.235 = **81.77%** of the full photo-to-preview
  run. Persistent Unreal will help native iterations, but cannot eliminate that
  fitting cost. See the measured manifest inventory in section 5.
- **Gap D - invalid work.** Photo-atlas skin and the brighter review preset were
  used in speaking drafts despite earlier native complexion and reel-lighting
  choices. Avoiding incorrect rebuilds is as important as faster execution.
  The current native-skin change skips Unreal photo-atlas baking/import and
  supports explicitly removing the photo override. It has not yet established
  final visual quality. Sources: `config\appearance-arka.json`,
  `config\reanimation.json`, `docs\facefitting\dev.readme.md`.

The existing subprocess design provides isolated failures and a strong fresh
reload boundary, but applies that expensive boundary too often. Preserve source
hashes, source geometry, requested complexion/grooms, camera/voice independence,
actual material inspection, native process errors and visual review. Never
trade those gates for startup savings.

This is a local repository, not a multi-team service. No Microsoft team,
on-call rotation or additional owner has been established; none is invented.

## 2. Implementation

### Verified execution choices

| Option | Evidence and limitation | Decision |
|---|---|---|
| Persistent editor plus Epic Python remote execution | Installed `remote_execution.py` exposes discovery, connection and repeated `run_command` calls. Official Python settings expose remote execution and host-local discovery. Needs a session lifecycle adapter and testing with asynchronous MRQ. | **Preferred after current generation.** Reuse the installed transport rather than build an HTTP server or custom MCP. |
| Offscreen editor / commandlet | Python documentation supports full-editor and commandlet script execution. Current D3D12/offscreen launches already avoid a visible window; each still initializes Unreal. MetaHuman conform, rig/texture requests and assembly require its editor subsystem. | Keep offscreen rendering. Do not equate no window with no engine startup, or assume commandlet/NullRHI supports every stage. |
| Runtime packaged renderer | Epic documents Movie Render Queue in runtime builds. That can render prepared content, but editor Python is editor-only; it does not replace the current editing/assembly path. | Later render-only option, not the first optimization. Cooking/build maintenance must justify its cost. |
| UELibrary embedding | The installed 5.8 header exposes initialize, tick, window messages and shutdown. Epic's accessible build tutorial is **4.27**, not a verified 5.8 MetaHuman recipe. Embedding still initializes an engine and would need custom builds/API exposure. | Not a drop-in standalone Python skin/rig/render library. Defer; no local embedding benchmark or MetaHuman compatibility claim. |
| Normal Python libraries | Existing fitting, image processing, manifests and FFmpeg orchestration already run outside Unreal. Native editor operations do not become ordinary Python functions by installing a package named `unreal`. | Keep portable work outside Unreal; do not rewrite Epic's native material/rig systems. |

Primary references: [Python scripting][python], [Python settings][settings],
[MetaHuman scripting][metahuman], [runtime MRQ][runtime], [UELibrary][uelibrary].
Installed API paths and exact inspected entry points are in section 5.

### Proposed lifecycle and responsibilities

**Host coordinator:** retain the current Python job APIs and JSON requests,
receipts, hashes and unique output namespaces. Add a dedicated Unreal-session
adapter rather than change the generic Blender/COLMAP subprocess behavior.
Start or explicitly attach to the intended project/editor; establish ownership
and engine/project/worker identity, not merely the first discovered editor.
Communicate with the editor through Epic's host-local UDP discovery and TCP
command channel. No remote execution settings or listeners were enabled during
this investigation. Source for transport capabilities: installed
`remote_execution.py:13-32,101-136`; [Python settings][settings].

**Persistent editor:** execute one mutating native operation at a time, keeping
the engine loaded between edits, assembly, review and render. Acknowledge an
asynchronous render submission separately from its final receipt; return
control to the engine tick loop while MRQ runs. Job completion must not invoke
editor shutdown. Keep Characters registered only while an edit session needs
them, then release preview actors, delegates and unused job state. Reusing an
editor alone does not remove a new Character's registration/synthesis cost.
Proposed changes concern `character_features_worker.py`, `unreal_worker.py` and
their launch adapters; they are not implemented by this research.

**Publication verifier:** preserve one fresh-process verification of the saved
Character/runtime after a completed asset revision, before final publication.
Ordinary adjustments do not each trigger it. A same-process unload/reload is
not equivalent evidence. Measure dual-editor RAM/VRAM before choosing whether
to retain the main editor during that check. If residency is too expensive,
use one deliberate checkpoint/restart and make the new editor the reusable
worker; do not silently weaken verification. The existing independent gate is
in `character_features.py:70-119`.

**Encoder/media verifier:** keep FFmpeg outside Unreal, reading the completed
PNG/WAV artifacts and producing a fully decoded, verified MP4. Do not spend
effort replacing a roughly two-second encode while startup and fitting dominate.
Sources: `src\ailias\reanimation\pipeline.py:111-129` and the manifests below.

Use the existing local JSON artifact model. No distributed database, broker or
distributed queue is proposed, so a CA/AP/CP classification would be misleading:
CAP concerns behavior under network partitions in distributed data systems.
The host admits at most one mutating operation; return an explicit busy result
rather than accumulate unbounded work.

Existing durable data relationships, not a proposed database schema:

```mermaid
erDiagram
    FEATURE_REVISION }o--o{ ARTIFACT : references
    REEL_JOB }o--o{ ARTIFACT : references
```

**Backed by:** `character_features.py:34-47,120-155`,
`reanimation\pipeline.py:54-64,111-129`. Multiple immutable manifests can
reference the same source artifact; blocked jobs may have no output assets.

### Full-pipeline priorities and measurements

| Priority / area | Measured baseline | Proposed action and acceptance constraint |
|---|---|---|
| P0: stop wrong rebuilds | Draft `bb05dc1c`: 743 decoded frames, 150.906s native job; actual skin/exposure rejected | Bind explicit appearance and reel policies to the reviewed candidate. Inspect a frame using the **same render/color pipeline and lighting** before the full clip. A technically playable clip cannot pass the skin/likeness gate. |
| P0: native startup/registration | Four operation launches for edit/build, verify, review and reel; recent registration 45.801s | Reuse the working editor; final independent verification remains. Compare cold startup, warm submission, registration, actual operation and shutdown separately. The registration sample is not a measurement of startup or an expected saving. |
| P0: unexplained startup stall | Revision `828334c0` verify process 2023.578s; worker 50.535s; observed log gap 1911.849s before worker execution | Capture CPU/asset-load/I/O startup traces after generation. Do not call the unexplained interval shader computation, cloud time or GPU work. |
| P1: fitting search | 162 candidates, 260 views, 3523.735s search / 4309.235s whole run | Profile existing CUDA operator/transfer/solve time on a bounded representative workload. Do not discard views, lower capacities or change winner selection merely for speed. Existing warm-start changes need their own real benchmark. |
| P1: observation extraction | 230.390s in the same full run | Split decode, detector, landmark, parser and transfer time first. Consider explicitly requested reuse keyed by image/model/preprocessing hashes; preserve fresh-run semantics and never silently replay earlier observations/fits. |
| P1: native asset build | Earlier cinematic apply 384.359s including 262.415s worker build | Rebuild only asset-affecting changes; distinguish appearance, geometry/rig, narration and camera/lighting invalidation. Do not assume a native complexion-index edit preserves downloaded texture validity. |
| P1: DDC / startup configuration | Cache hits, storage latency and Developer Mode are **not measured** | Inspect existing local Derived Data Cache before changing it. Epic documents DDC reuse and Developer Mode's boot-time stub generation. Do not delete caches, migrate the shared machine or disable needed plugins speculatively. |
| P2: Blender | 59.094s, one export/review stage in the full run | Retain hair-free Blender and required fresh geometry export. Do not restart fitting/Blender for an appearance-only Unreal iteration. |
| P2: native photo atlas | Bake 15.344s plus import 26.953s | Current configured native-complexion route skips these Unreal-specific stages; source Blender appearance remains separate. These totals include real work, not just removable launch overhead. |
| P2: rendering/encoding | Earlier 743-frame MRQ 103.326s; encode about 2s | Reuse stage/sequence/solved Performance only when dependencies and explicit reuse permit. Keep 540x960 iterative resolution and the established camera/lighting; no lower-quality final claim. |

Sources: measured manifests in section 5; code paths above; [DDC][ddc],
[Unreal Insights][insights], [PyTorch profiler][profiler]. These are priorities,
not measured speedups. No benchmark or new profiling capture was run during the
active generation.

**Scale and throughput:** the observed workload is one local owner/project with
260 fitting views, 162 search configurations and a 743-frame draft. At that
observed full-run duration, `1 / 4309.235 = 0.000232` full runs/second; the
150.906s native reel stage is `1 / 150.906 = 0.00663` stages/second, excluding
encoding and other pipeline stages. These are single-run reciprocals, not
capacity tests. Proposed admission is one active GPU/editor operation, no
unbounded backlog; peak read/write request rates and sustained disk/VRAM usage
remain to be measured. A candidate warm session changing four launches to two
has a **model**, not a forecast, of `2 * startup_cost - resident_session_overhead`
saved; cloud, synthesis, registration and actual rendering remain.

**Backpressure and single points of failure:**

| Critical component | Failure / pressure | Mitigation in the proposal |
|---|---|---|
| One editor / project writer | Stuck native operation, crash or conflicting mutations | One active operation, job deadline, durable receipts/checkpoints, explicit busy response, restart only owned process |
| One local GPU | Fit/render contention and unknown dual-editor VRAM peak | Serialize heavy work by default; measure residency before overlapping verification; no unrelated process termination |
| Epic rig/texture services | Network/authentication/service wait cannot usefully be solved by local CUDA | Stage-specific progress/error evidence, preserved prior assets, no automatic replay of uncertain completed requests |
| Local disk / DDC | Large assets, PNG writes, cache miss or slow storage | Measure I/O/cache behavior, retain verified checkpoints, bounded job-owned cleanup; do not erase shared caches |
| Persistent process state | Stale objects, delegates, MRQ globals, leaks across jobs | Lifecycle/failure tests, per-job namespace and final independent saved-asset check; controlled recycle only on evidence |

### Targeted diagnostics and files

Use stage-duration stacked bars to separate startup/registration/cloud/build/
render; warm-latency distributions across repeated operations; RAM/VRAM max
time series for residency; and a job/event timeline for stalls. Do not add a
dashboard service for this single-machine workload. Existing stage receipts are
the initial source; missing spans require bounded local instrumentation.

Keep the current renderer/material gates and source-integrity checks. Proposed
edits belong in the Unreal-specific launcher/session adapter and native worker
lifetime callbacks; image/CUDA optimizations belong in the existing preparation
and fitting modules after a profile identifies a real cost. New public session
commands/configuration, if introduced, require documentation and explicit
ownership-aware start/status/stop semantics, not an automatic global daemon.

### Decided code changes

This is the **draft edit plan**, not a list of changes already implemented.
Optimize measured wall time subject to unchanged output quality, source
integrity and resource limits; "absolutely optimized" is not a result that can
be asserted without benchmarks. Finish/diagnose the active visual iteration
before migrating execution.

**Latest measured skin iteration:** revision `f07f4c69` took
`279.688 + 83.562 + 33.312 = 396.562s` (6m37s) for apply/build, independent
verification and preview. Within apply, committing native skin settings took
**0.01454s**, Character registration **46.081s**, runtime assembly **166.579s**,
and package/Character saves about **35.9s**. Verification registered the
Character again for **48.333s**. No cloud request or rerig occurred.
Its actual review `62dc53f6` still failed with checkerboard skin; empty-override
cleanup did not solve it. These are end-to-end/contained spans, not independent
durations to add again. Sources:
`data\arka\reanimation\native-skin-corrections\68b63f25e69f41a0842d2aee152120b8\manifest.json`
and its referenced feature receipts.

#### A. Execution lifecycle: reuse one working editor

| Existing file / proposed new file | Planned edit | Required test |
|---|---|---|
| **NEW** `src\ailias\generation\unreal_session.py` | Implement an owned/attached `UnrealSession` using installed Epic `remote_execution.py`. Expose `start`, `attach`, `submit`, `status`, `wait`, `close`. Keep engine/project identity, PID plus creation time, request ID, deadline, and bounded receipts. Import no engine or remote transport on unrelated commands. | Fake transport/process tests: two submissions reuse one PID; wrong project/engine rejected; dead/reused PID not treated as an editor; attach does not grant shutdown ownership. |
| **NEW** `src\ailias\generation\unreal_session_worker.py` | Bootstrap the long-lived embedded-Python dispatcher. Execute a hash-pinned known worker with an explicit request dictionary, not per-job environment mutations. One active native mutation; acknowledge asynchronous submission and finish via job-specific callbacks. | Accept/busy/complete/fail/cancel transitions; stale callback cannot finish a later job; different worker hashes cannot reuse stale module state. |
| **NEW** `config\unreal-session.json` | One validated source for execution mode and session startup/connection limits. Existing fitting/reel policies retain operation timeouts, devices, lighting and quality. During rollout keep legacy one-shot mode; switch the personal workflow to persistent only after the real acceptance run. | Invalid settings fail before process launch; explicit attach failure never silently launches another editor. No engine paths hardcoded into source. |
| `src\ailias\generation\cli.py` | Add proposed `generation unreal-session start/status/stop` actions. Handle them before generic reconstruction policy/loading. Start is explicit and keeps the owned editor available across CLI invocations; status never starts it; stop targets only its verified owned PID. | CLI actions do not probe COLMAP/Blender, import models, or close an attached/user-owned editor. Commands are proposals until implemented/documented. |
| `src\ailias\facefitting\cli.py`; `src\ailias\reanimation\cli.py` | Accept an explicit session ID/config and pass one session object through the requested workflow. Command-line session choice wins over session config; existing appearance/reel configuration precedence is unchanged. | Existing invocations/defaults remain compatible; a supplied session cannot silently select a different project or policy. |
| `src\ailias\generation\tools.py` | **Do not turn every native tool into a daemon.** Keep current subprocess runner for Blender/COLMAP and explicit cold/verifier runs. Reuse its bounded log/error behavior for the session adapter where applicable. | Existing native-tool failure/timeout tests continue to pass; unrelated tools retain their current lifecycle. |

Use the existing artifact store, not a database or new broker. Proposed session
ownership records live under the configured data root's
`.runtime\unreal\PROJECT_FINGERPRINT`; distinguish a host-local control record
from an alias's artifact history. `Store.lock` is alias-scoped
(`storage.py:47`), so the session adapter must additionally serialize all writers
to the **same Unreal project**, including different aliases. Do not silently
change the existing store lock's semantics.

Keep the editor ready after the Unreal handoff; do not boot it early merely to
occupy VRAM throughout a large CUDA fit. Measure GPU/RAM residency before
overlapping heavy stages. No speculative multi-editor parallelism or automatic
termination of other users' applications.

#### B. Worker migration and final verification boundary

| Existing file | Planned edit | Boundary that must remain true |
|---|---|---|
| `facefitting\unreal\worker_process.py` | Add an optional session execution route to `execute_unreal_worker`; retain current request/receipt schema, explicit status map and error propagation. | A transport acknowledgement is not operation completion; missing/wrong/failed native receipts still fail. |
| `facefitting\unreal\character_features.py` | Extract independent verification so a session can make explicitly labelled warm **drafts**, then finalize one saved asset revision with a fresh process. Preserve today's immediate verification as the compatibility default. | Draft reports must not set `fresh_editor_reload_verified=True` or claim the existing publication-success status. |
| `facefitting\unreal\character_features_worker.py` | Reuse `run(request, unreal, publish)`; pass session/job context without relying on process exit. Coalesce only provably unchanged edits. Before another assembly, diagnose actual native synthesized textures versus baked runtime textures/materials. | Keep native property, source geometry, actual LOD skin/groom and package-save checks. Native complexion-index changes can invalidate texture sources. No photo override to hide a native bake failure. |
| `facefitting\unreal\character_portrait.py` | Support an explicitly labelled unverified **draft** review from the working session; keep verified-source behavior as default. Reuse the same MRQ render/color pipeline and explicitly selected reel policy when assessing a future reel. | A draft review cannot bypass the final independent gate or silently inherit the brighter preview exposure. |
| `reanimation\unreal_worker.py` | Separate execute/stage/render from CLI bootstrap and editor shutdown. Replace `REQUEST/RESULT/EXECUTOR/QUEUE` ownership with request-scoped render state; retain required callbacks/objects until completion, then remove only that job's delegates/actors. | `finished` must complete a job, not terminate a persistent editor; no cross-job callback/global-state leakage. Existing one-shot script exit remains supported. |
| `reanimation\pipeline.py` | Inject the session, preserve media gates, and record the actual verified source revision/dependencies as well as Blueprint path. Gate full render on the same-source/same-policy visual review. | Unchanged voice, camera/lighting, exact frame range, all-stream decode, hashes, native failures and source isolation. |
| `facefitting\workflows\photo_to_character.py` | Pass the same session through native conform/appearance/review. For the new native-skin route, consolidate independent geometry/appearance/runtime checks at the final asset boundary only after the existing equivalence tests prove nothing was omitted. | Required fresh geometry and zero-drift proof still happen before publication. Keep the legacy photo-projection route on its current boundaries until separately validated. |
| `facefitting\workflows\video_character_refinement.py`; `facefitting\unreal\sculpt_response.py` and `_worker.py` | Migrate the alternate native-refinement path after the main path passes. Replace worker-owned unconditional `quit_editor` with explicit lifetime ownership. | No direct process exit inside a shared edit/calibration job; preserve actual mesh/topology checks and owner-requested frozen geometry. |

Paths in the table are relative to `src\ailias` unless the full `src` prefix is
shown. These call sites and lifecycle behavior were directly inspected; new
session files and APIs are proposed. Do not parse arbitrary historical command
lines to infer an operation: launch adapters submit explicit structured jobs.

The **working editor remains open** across ordinary operations. One clean
verifier remains at final saved-asset publication, not after every small edit,
not for every camera adjustment, and not represented as a warm reload. If a
second resident editor exceeds measured memory budgets, use one deliberate
checkpoint/restart and reuse the restarted editor; this is an explicit
resource-bound fallback, not the current per-stage restart loop.

#### C. Dependency rules: do not run unrelated stages

| Change requested | Recompute / validate | Must not automatically repeat |
|---|---|---|
| Inspect properties or compare settings | Read current receipt and native state; detect no-op edits | Full fit, Blender, rig, texture download, assembly or full video |
| Camera, exposure, light, framing, resolution or backdrop | Re-stage/re-render using explicit reel policy | Identity fitting, texture generation, rigging, grooms or audio solve |
| Narration / solver emotion settings | Speech performance and final sequence/render against the verified target | Geometry fitting, native skin synthesis or character assembly |
| Groom choice | Native wardrobe/binding steps required by that selection; actual deformation/LOD checks | Photo analysis or identity refit |
| Native complexion / texture index | Inspect whether native source textures/material synthesis actually need invalidation; synthesize/bake only the affected required assets | Geometry refit, Blender or rerig unless native prerequisites explicitly prove one is needed |
| Photo/evidence/model/fit policy | Required observation/fitting/native geometry descendants under a new fresh run | Silent reuse of old fits, observations or generated outputs |
| Explicit reuse of a saved Performance | Verify narration, solver policy, target rig/DNA/control compatibility and asset hashes; render a new requested output | Re-solve the same explicitly selected performance or reuse unrelated voice/target data |

Implement these rules first as explicit workflow boundaries, **not** a new
general DAG/workflow framework. Extend existing manifests with dependency and
verification evidence. Propose `--performance-job JOB` in
`reanimation\cli.py`/`pipeline.py` only when implementing that opt-in path;
ordinary retry still means fresh generation, not silent output caching.
`reanimation\media.py` keeps its full frame/audio/video verification.

#### D. Deterministic fitting work reduction, then profiling

There is a concrete redundant-solve candidate in
`facefitting\geometry\identity_search.py:79-101`: zero-identity training and
validation reference fits are recomputed inside the identity-prior loop.
When identity has zero modes, changing **only identity regularization** adds
no identity residual. The expression model, expression prior/bound, observations,
camera policy and solver configuration must all still match.

- **Planned first edit:** share those reference pairs within the same search
  run across identity-prior values, keyed on all effective nonidentity inputs.
  At the current configured dimensions, `3 expression capacities * 3 identity
  priors * 3 expression priors * 2 bounds * 2 reference fits = 108` reference
  calls could become `3 * 3 * 2 * 2 = 36`: **72 redundant reference solves
  avoided**, not 72 identity candidates removed. This is a code-derived
  opportunity, **not a measured speedup**. Verify that solver/model/view state
  is not mutated before introducing reuse.
- **Unchanged:** all 162 candidate records, capacities, training starts,
  observations, held-out scoring and final all-view fit. An initialization
  optimization is not permission to choose candidates using held-out data.
- **Next edit only if profiled:** prepare immutable model/view tensors once
  per compatible run/model/device in `geometry\cuda_optimization.py` and route
  that optional prepared context through `geometry\optimization.py`.
  Current `fit_cuda` constructs/transfers those tensors per call. Measure
  preparation versus Jacobian/linear-solve cost before adding a cache/context.
- **Further conditional work:** profile the necessary scalar synchronizations,
  Jacobian batch size and allocation peaks in `cuda_optimization.py`; do not
  remove convergence checks, reduce views/capacities or silently enable mixed
  precision/TF32/compilation. Preserve the existing working CUDA installation.
- **Preparation:** instrument `facefitting\evidence\photo_processing.py`
  before changing detector/landmark/parser scheduling. If evidence supports
  explicit observation reuse later, key it by original bytes, model revision
  and preprocessing; preserve the current fresh-prediction contract by default.
- **I/O:** measure repeated artifact hashing before optimizing it. If material,
  add boundary-local deduplication to `storage.py`; never use a global
  modification-time-only hash cache or omit final source verification.

Tests belong in existing `geometry\tests\test_identity_search.py`,
`test_batched_optimization.py`, `test_optimization.py`, workflow tests and
imageprocessing evidence tests as applicable. Compare selected candidates,
training/held-out metrics, failed-start handling and invalidation under changed
expression limits/models. Do not test speed using a wall-clock assertion in
mock-based units.

#### E. Tests, rollout and concrete acceptance

**New tests:** `src\ailias\generation\tests\test_unreal_session.py` and
`test_unreal_session_worker.py` (proposed), plus native opt-in coverage under
`src\integration_tests\test_unreal_session.py` (proposed). Unit tests use fake
transport/processes and introduce no engine/model imports. Real engine/GPU
tests remain explicitly selected, not part of rapid-development discovery.

| Gate | Exact evidence required |
|---|---|
| Compatibility | Existing tool, feature, workflow, portrait and reel units pass unchanged before enabling persistent mode; no new dependencies or package upgrades required for the proposed Epic transport. |
| Reuse | A cold start followed by at least three warm operations retains the same working PID/project and does not repeat engine startup. Record full wall time and nested spans; do not subtract unrelated samples. |
| Correctness | Same geometry, explicit native complexion/grooms, camera, tone pipeline, exposure, voice and frame-count/decode gates as the approved baseline; actual new frames inspected. Current checkerboard skin remains a failure. |
| Isolation | Different aliases targeting one project cannot mutate concurrently; wrong session/project rejected; borrowed editor left running; timeout/cancel/disconnect cannot publish a stale result. |
| Async lifecycle | Two sequential MRQ jobs complete without stopping the editor, leaking old delegates or assigning a late completion to the next job. A blocking cloud job reports cancellation pending if it cannot be interrupted safely. |
| Recovery | Crash leaves explicit failed/unknown job state and prior saved artifacts intact. No automatic replay of an uncertain cloud/generation request or stale successful receipt. |
| Resources | Record peak RAM/VRAM and object/delegate counts across warm operations; recycle only on a demonstrated leak/error or explicit resource budget, not after every job. |
| Search reduction | Prove effective-input equivalence and unchanged candidate/score results for the zero-mode reference reuse; then separately benchmark on the real configured GPU without changing the search. |
| Publication | One independent verifier per finalized asset revision; previews stay drafts until that check and visual review pass. No verifier launch for an unchanged already-verified asset solely because lighting changed. |

**Implementation order:** native texture/material diagnosis without further
blind full builds; OPT-1 instrumentation; minimal OPT-2 transport/lifetime proof;
main native/reel migration and finalization gate; dependency-aware iterations;
deterministic reference-solve reduction; then profile-selected preparation/GPU/
I/O improvements. Each functional slice gets targeted tests before expansion.
Keep one-shot fallback explicit until the full real acceptance run passes.
Update `docs\reanimation\README.md`, `dev.readme.md`,
`docs\facefitting\README.md`, `docs\analysis.md` and the root command index
when the corresponding behavior is actually implemented, not before.

## 3. Success Criteria

| Criterion | Baseline | Target / guardrail |
|---|---|---|
| Process reuse | Per-operation launches | Two sequential appearance/review operations execute in the same owned PID; preview and reel do not restart the working editor |
| Saved-asset correctness | Independent Character/runtime reload | Preserve one clean verification per finalized asset revision and actual per-LOD skin/groom evidence |
| Latency | Timings above; warm-session baseline absent | Measure cold and warm runs with identical inputs/settings; report absolute/relative improvements only after measurement |
| Resident resources | Peak not measured | Bounded actors/delegates/RAM/VRAM across repeated jobs; no unrelated editor/process shutdown |
| Fault handling | Subprocess timeout and native receipt gates | Inject busy, disconnect, late callback, native failure, crash and cancellation; no stale success or duplicate publication |
| Appearance/reel policy | Prior native choices and -5.5 EV video preset exist | Preserve requested geometry, native complexion, hair, lighting, framing and voice; inspect actual new frames and reject regressions |
| Search quality | All 162 candidates and 260 views accounted for | No silent evidence/capacity loss, changed holdout selection or output replay |

Exit requires a real two-job warm-session test, final independent reload and
actual new rendered frame/video review, with failures and resource limits
reported. Documentation/API availability alone does not establish feasibility
of our complete asynchronous MetaHuman/MRQ lifecycle.

## 4. Task Breakdown

Sizing is a rough work-item estimate including tests/hardening, **not an ETA for
the current video**. No orchestration refactor begins during active generation.

| ID | Deliverable | Dependencies / estimated size |
|---|---|---|
| OPT-1 | Instrument cold/warm startup, Character registration, cloud, native build, render and encoding; diagnose the long pre-worker stall using Insights and DDC evidence | Gaps A/C; about 1 week for a reproducible profiling/benchmark suite |
| OPT-2 | Add the owned persistent-editor adapter using installed Epic remote execution; separate worker completion from editor lifetime; test async MRQ, clean cancellation and final verification | Gap A/B; OPT-1 baseline; about 1-2 weeks including real two-job and leak/failure validation |
| OPT-3 | Add dependency-aware native/reel iteration boundaries and profile-guided fitting/preparation improvements, preserving fresh-generation semantics and all visual/quality gates | Gap C/D; OPT-1, and OPT-2 for resident-session acceptance; about 1-2 weeks |

## 5. Reference Links

**Repository source:** paths and function locations above. Historical timing
evidence lives under `data\arka`, not in a public telemetry dashboard:

- `facefitting\fresh-video\252acd173452429388a7e40718bce9a6\manifest.json`:
  `seconds`, `stages`, and native operation runtimes.
- `facefitting\features\828334c0592049a982a24076f4b623c6\manifest.json`:
  apply/verify runtime; corresponding verify log and receipt separate process
  wait from actual worker time. Long startup gap cause remains unestablished.
- `facefitting\features\185f51210d0e481096d42040f8c28e84\apply-receipt.json`:
  observed 45.801s registration sample during the current correction; not a
  completed-generation claim.
- `reanimation\runs\639dc1bc90234c55b2c4ecacd14a27d6`:
  earlier measured MRQ/encoding logs; rejected checkerboard-skin draft.
- `reanimation\runs\bb05dc1cfd5842cb9da2912498eca2a6\manifest.json`:
  native runtime, decode evidence, policy and actual frame references; visually
  rejected photo-atlas/bright-exposure draft.

**Installed Epic APIs inspected read-only:**

- `C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python\remote_execution.py`:
  protocol/defaults at lines 13-32; connection/command API at 101-136.
- `C:\Program Files\Epic Games\UE_5.8\Engine\Source\Runtime\UELibrary\Public\UELibraryAPI.h`:
  initialization, tick, window messages and shutdown; no direct MetaHuman
  fitting/skin/assembly API is exposed by this header.

**Official documentation checked November 18, 2026:**

- [Editor Python scripting][python] (5.8).
- [Python settings and remote execution][settings] (versioned page currently
  redirects to the current documentation; installed 5.8 transport checked too).
- [MetaHuman Creator scripting][metahuman] (5.8+ prerequisites).
- [Movie Render Queue in runtime builds][runtime].
- [UELibrary build tutorial][uelibrary] (**4.27**; 5.8 embedding untested).
- [Derived Data Cache][ddc].
- [Unreal Insights][insights] (current reference; installed trace support must
  be checked before configuring a 5.8 capture).
- [PyTorch profiler][profiler] (current docs; use only APIs supported by the
  existing pinned environment, no package upgrade implied).

Still required: cold/warm traces, peak memory measurements, installed project
remote-execution configuration verification and a real persistent async render
proof. No external team aliases, dashboard, PR or standalone MetaHuman SDK
benchmark was established.

**Generation outcome during research:** native revision `185f5121` completed
application and independent reload in 317.859s/129.109s. Its actual review
`64beeee4` still shows gray checkerboard face/body skin despite correct saved
native controls and restored -5.5 EV exposure. It was rejected; no full video
was started. This demonstrates why warm execution must retain actual texture
and visual gates rather than treating a successful receipt as acceptable skin.

[python]: https://dev.epicgames.com/documentation/en-us/unreal-engine/scripting-the-unreal-editor-using-python?application_version=5.8
[settings]: https://dev.epicgames.com/documentation/en-us/unreal-engine/python-settings-in-the-unreal-engine-project-settings
[metahuman]: https://dev.epicgames.com/documentation/metahuman/metahuman-creator-python-scripting-in-unreal-engine?lang=en-US
[runtime]: https://dev.epicgames.com/documentation/unreal-engine/movie-render-queue-in-runtime-in-unreal-engine?lang=en-US
[uelibrary]: https://dev.epicgames.com/documentation/en-us/unreal-engine/building-unreal-engine-as-a-library?application_version=4.27
[ddc]: https://dev.epicgames.com/documentation/unreal-engine/using-derived-data-cache-in-unreal-engine?lang=en-US
[insights]: https://dev.epicgames.com/documentation/unreal-engine/unreal-insights-reference-in-unreal-engine-5
[profiler]: https://docs.pytorch.org/docs/stable/profiler.html

[Back to REPO_ROOT](../../README.md) | [Native execution](../generation/dev.readme.md)

# Native execution and dependency boundaries

## 1. Problem Statement

Native Character editing, assembly and rendering need an initialized Unreal
editor. Repeated process startup is avoidable; independent saved-asset
verification and material/media correctness are not.
Offscreen execution removes a visible window, not the editor runtime.

The local pipeline must preserve source geometry, requested appearance,
camera/voice independence, native failure reporting and visual-review boundaries.
No database, remote execution service or distributed queue is required.

## 2. Implementation

### Execution choices

| Choice | Why chosen | Tradeoff |
|---|---|---|
| Owned persistent editor and local JSON mailbox | Reuses existing request/receipt contracts without sockets or discovery | Resident memory and state require explicit ownership and lifecycle checks |
| Independent verification process | Checks what was actually saved, not only live editor objects | Extra process cost per verified revision |
| Explicit one-shot mode | Isolated lifecycle for supported operations and cold checks | Does not remove startup cost |
| Portable host Python | Keeps fitting, evidence, storage and FFmpeg outside Unreal | Native editor APIs still require embedded Python |
| No remote listener | Single-host file transport is sufficient | Does not provide remote job submission |

`generation\unreal_session.py` validates project/bootstrap identity and PID
creation time, then admits one request under the project OS lock.
`unreal_session_worker.py` runs a hash-pinned module on the editor thread.
Pending requests cannot be overwritten. Host timeout does not cancel or replay
native work; a failed/uncertain session requires inspection before reuse.

The dispatch callback is removed during synchronous native work because native
assembly can pump Slate, Unreal's UI/event loop. It is restored after execution.
Asynchronous render jobs retain their module and callbacks until a terminal
receipt, then release only their own resources. Job console variables are
restored before admission reopens. Jobs do not stop the shared editor.

### Correctness boundaries

Migrated launches explicitly select D3D12/SM6 and verify actual shader-platform
metadata. Material compilation failures reject publication even after successful
exports. Only precisely identified standalone function-thumbnail preview
warnings are excluded; actual runtime/master materials remain checked.
Never conceal an invalid native material with photo-colour fill.

Original sources, saved packages and narration retain content hashes.
Independent reload checks remain separate from mutation. Complete frame/media
validation remains separate from render callback success.

### Operation dependencies

| Requested operation | Required work | Unrelated work |
|---|---|---|
| Read status/properties | Inspect current state and references | No fit, rig, assembly or render |
| Neutral portrait review | Verify runtime and render with explicit review policy | No speech solve or identity fit |
| Narrated reel | Fresh audio solve, sequence/render and media validation | No identity fit or Character creation |
| Feature edit | Validate profile, duplicate/save/reload; optional requested assembly | No automatic photo refit |
| Fresh photo-to-character run | Fresh observations, fit, Blender handoff and native descendants | No silent replay of old predictions/results |
| Major-output cleanup | Inspect references/native registry and revalidate deletion plan | No refit, render or editor shutdown |

Explicit reuse of a saved audio Performance is an extension, not current
`reanimation run` behavior. Ordinary retry still creates fresh generated output.
Do not present a proposed dependency optimization as an existing resume cache.

Within an explicitly requested identity search, compatible zero-identity
reference solutions are reused by effective expression model/prior/bound.
Candidate evaluation and final fitting remain unchanged. Checksum reuse is
limited to one inspection and invalidates on file identity changes.

### Capacity and failure

One project session admits one job; alias locks independently exclude cooperating
alias writers. There is no unbounded queue or global GPU scheduler.
Native inference/compile/render, GPU memory, frame writes and encoding can apply
backpressure. The editor, GPU and local disk are local single points of failure;
retain diagnostics and immutable revisions rather than claim high availability.

Native retention uses registry inspection without loading old assets. Eligible
files remain queued while Unreal is open. Owned-editor exit invokes offline
cleanup only after full snapshot/dependency/ownership revalidation.

## 3. Success Criteria

| Gate | Required evidence |
|---|---|
| Ownership | Wrong project/bootstrap/recycled PID rejected; unrelated editors untouched |
| Reuse | Sequential jobs use the same owned editor without routine restarts |
| Async isolation | Late callbacks cannot complete a different job; delegates are removed |
| Timeout | Unconfirmed work is not replayed or cleaned while still writing |
| Saved assets | Independent reload verifies requested state and unchanged sources |
| Materials | Actual shader platform and runtime compilation checks pass |
| Media | Exact frames and full audio/video decode pass |
| Optimization | Equivalent inputs/candidates/gates before any measured speedup claim |

## 4. Task Breakdown

| Change area | Validation procedure |
|---|---|
| Session lifecycle | Fake-process ownership, exclusion, stale receipt and failure tests |
| Worker lifetime | Sequential request/callback tests; explicit native smoke integration |
| Search work | Reference reuse/invalidation and failed-reference recovery tests |
| Retention | Metadata races, live dependencies, package snapshot and active-writer tests |
| Further optimization | Profile a specific phase; retain unchanged correctness gates |

Commands and opt-in integrations are in
[generation internals](../generation/dev.readme.md#tests).
Execution histories and measured timings are kept outside the manuals.

## 5. Reference Links

- [Local native execution contract](../generation/dev.readme.md).
- [Public session commands](../generation/README.md#reusable-unreal-editor).
- [Epic: editor Python](https://dev.epicgames.com/documentation/en-us/unreal-engine/scripting-the-unreal-editor-using-python).
- [Epic: MetaHuman Creator scripting](https://dev.epicgames.com/documentation/metahuman/metahuman-creator-python-scripting-in-unreal-engine?lang=en-US).
- [Epic: command-line arguments](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-command-line-arguments-reference?application_version=5.8).

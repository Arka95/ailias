[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Workflow implementation

## Ownership

| Module | Responsibility |
|---|---|
| `contracts.py` | `ABCMeta` workflow/action contracts and optional resource-free hooks |
| `context.py` | Current/named shared values, exclusive run-directory reservation and explicit JSON snapshots |
| `workflow.py` | GUID allocation, sequential execution and lifecycle dispatch |
| `common\json_values.py` | Domain-independent conversion of structured values to strict JSON-compatible data |
| `storage.py` | Existing atomic JSON replacement and temporary-write cleanup |

The runtime imports no image-processing, Blender, Unreal or model SDK modules.
Domain-specific actions adapt their own APIs to the shared context. Existing
`ExecutionChain`/`StageJournal` callers remain unchanged: those helpers have
their established immediate-cleanup semantics, while this API requires explicit
save/cleanup calls. The manual handoff module `ailias.workflow` is unrelated.

## Execution and persistence

The constructor consumes the declared actions, validates `IAction` membership,
generates a UUID and constructs context without filesystem output. Execution or
saving reserves `<directory>\<UUID>` exclusively; an existing run is never adopted.
`artifacts` and `scratch` are conventions exposed as context paths, not action IDs.

The executor registers cleanup before entering each action. Only a successful
return replaces `.current` and registers that action for the save/export hook.
Shared-state mutations or external side effects inside a failed action are not
rolled back. No implicit retries, scheduling, concurrency, inference or recovery
are introduced. Another execution requires another workflow instance/run GUID.

`Workflow.save()` invokes completed actions' export hooks in order, then delegates
to `WorkflowContext.save()`. All hooks see current shared state and may export
owned assets or add references; no per-action result history is invented.
`to_json_value()` prefers `to_dict()`, then supports ordinary dataclasses, paths
and UUIDs. Strict JSON conversion happens before the existing atomic writer runs.
Opaque values are errors, never arbitrary object strings or embedded binary data.
File replacement is atomic; a sequence of custom asset-export hooks is **not**
a transaction. Restoring Python classes or resuming a saved run is not implemented.

## Cleanup and errors

Cleanup visits entered actions in reverse order, including interrupted/failed
execution. Successfully cleaned hooks are removed from the pending list; failed
hooks remain retryable. Every hook is attempted, with aggregate failure reporting
unless an active primary exception must be preserved through exception notes.
This includes ordinary execution errors and interruptions.

Action implementations own their resources and scratch cleanup. This generic
runtime does not invent an alias-wide retention policy or delete an unknown file
tree; domain adapters should use existing resource-management APIs where applicable.
Saved JSON/artifacts and caller-owned resources must survive cleanup. An extractor
passed into an action remains caller-owned unless ownership was explicitly transferred.
The workflow retains its context for inspection but rejects export after cleanup,
when action resources may already be released.

## Validation

```powershell
python -m unittest ailias.workflows.tests.test_workflow ailias.common.tests.test_execution -q
python -m ruff check src\ailias\workflows src\ailias\common\json_values.py
```

Units use temporary files, fake atomic actions and in-memory feature objects:
chain handoff, FaceFeatures snapshots, current-state replacement, GUID isolation,
failure/interrupt propagation, reverse cleanup/retries, input/output preservation,
serialization/export failures and existing-run collision protection. No real
models, external services, subprocesses or native editors run.

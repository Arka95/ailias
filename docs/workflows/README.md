[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Workflow objects

`ailias.workflows` provides independent, single-run objects for chaining atomic
actions. It does not replace the existing manual `workflow.py`, change CLI
commands, migrate existing jobs or run any models automatically.

| Object | Responsibility |
|---|---|
| `IWorkflow` | `ABCMeta` interface requiring a run GUID plus `execute()`, `save()` and `clean()` |
| `Workflow` | Run an ordered action chain, save current context and delegate owned-resource cleanup |
| `IAction` | Atomic `execute(context)`, `save(context)` and `clean(context)` contract; no action GUID |
| `Action` | Abstract execution base with optional no-op export/cleanup hooks for resource-free operations |
| `WorkflowContext` | Latest output in `.current`, named values in `.data`, read-only run identity/directory |

## Usage

Define domain actions around existing APIs. This example assumes `prepared`
already came from image preparation; the caller owns the extractor's lifetime:

```python
from pathlib import Path
from ailias.imageprocessing.face_feature_extraction import FaceFeatureExtractor
from ailias.workflows import Action, Workflow

class ExtractFeatures(Action):
    def __init__(self, extractor):
        self.extractor = extractor

    def execute(self, context):
        return self.extractor.extract_features(context.current)

with FaceFeatureExtractor(mediapipe_model=Path(r"data\models\face_landmarker.task")) as extractor:
    job = Workflow(
        [ExtractFeatures(extractor)],
        directory=Path(r"data\workflows"),
        initial=prepared,
    )
    try:
        face_features = job.execute()
        snapshot = job.save()
    finally:
        job.clean()
```

Add more actions to the list to chain operations. Each successful return becomes
`context.current` for the next action. Use `context.data["face_features"]` for
named evidence that must remain available after later actions change `.current`.
Actions can also exchange file references using the conventional run directories.
“Atomic” means one logical operation, not transactional rollback of arbitrary effects.

## Lifecycle and output

`directory` is an explicit output root; each instance generates its own UUID/GUID:

```text
data\workflows\<run-guid>\
    context.json
    artifacts\
    scratch\
```

- `execute()` runs once and stops on the original failure. It does not save or clean automatically.
- `save()` calls completed actions' export hooks, then atomically writes the **current** context to `context.json`. It may also save initial/partial state without replaying execution.
- `clean()` calls entered actions, including a failed action, in reverse order. Successful hooks run once; failed cleanup hooks can be retried.

The JSON contains `schema_version`, `run_id`, `current` and `data`. Structured
objects such as `FaceFeatures` use their `to_dict()` contracts. Nested dataclasses,
paths and UUIDs are supported. Images, model handles and other opaque runtime
objects require an explicit serialization/export adapter; they are not silently
stringified. Unsupported/nonfinite/cyclic values leave the last valid snapshot intact.
This is a data snapshot, not an automatic object-restoration or resume mechanism.

An action's `save(context)` hook exports any separate assets and can put their
references in shared state; the workflow writes the JSON snapshot afterward.
For pure transformations the default hook does nothing beyond this workflow-level persistence.
All save hooks see the latest shared context, not historical per-action input snapshots.

Put durable exports in `context.artifacts_directory` and temporary files in
`context.scratch_directory`. An action must clean only its own handles/files and
preserve original inputs and durable outputs. The framework never recursively
deletes a run or blindly empties scratch. Context data remains inspectable after
cleanup, but save through the workflow **before** cleaning resource-owning actions.

Cleanup attempts every entered action even if another hook fails. Without an
active primary exception, failures are reported together; inside a caller's
`finally`/exception handler they are attached as notes to the active exception.
Instances run serially; create a new workflow for another run. No CLI or MCP
surface and no head-reconstruction adapter are included in this foundation.

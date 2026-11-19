[Back to REPO_ROOT](../README.md) | [Core workflow](workflow.md)

# Development

## Setup and tests

Use Python 3.12+ in the shared `C:\projectWorkspace\.venv` environment.
Dependencies and optional extras are declared in `pyproject.toml`.
Do not replace an existing CUDA/OpenCV/Torch stack to install an optional extra.

```powershell
python -m pip install -e ".[dev]"
python -m ruff check src\ailias --exclude tests
python -m unittest discover -s src\ailias -t src -q
```

Unit tests use temporary inputs and fake model/native adapters. Real model
execution, GPU checks and editor integration are explicit operations under
`src\integration_tests`, outside ordinary discovery. Run focused module tests
when changing one domain; use the complete suite for cross-module contracts.

## Architecture

The [module diagram](../README.md#module-architecture) shows data flow.
The host makes synchronous Python calls into domain modules. Blender, COLMAP
and FFmpeg use owned subprocesses with file inputs and bounded logs. Migrated
Unreal operations submit request files to one project-owned editor; asynchronous
renders complete through native callbacks and job-specific receipts. Independent
saved-asset checks use a separate editor process.

| Component | Responsibility and communication |
|---|---|
| `cli.py`, module CLIs | Validate arguments, call Python APIs and emit JSON results |
| `pipelines` | Compose cross-domain operations using `common.execution`; persist stage boundaries |
| `imageprocessing` | Decode, group, mask and measure images through task-scoped model adapters |
| `head_reconstruction` | Fit shared shape/per-photo cameras, export meshes and prepare photo colour |
| `metahuman` | Import/conform native geometry, apply feature profiles and verify saved assets |
| `generation` | Alternative photogrammetry plus shared native-tool and Unreal-session adapters |
| `reanimation` | Solve supplied audio on an existing rig, render frames and validate encoded media |
| `storage.py`, `workflow.py` | Alias locks, atomic JSON, artifact identity and manual handoff state |
| `resource_management` | Finalize owned scratch; plan and apply reference-aware retention |
| `common` | Domain-independent execution, JSON traversal, identifiers, source lookup and checksums |

`facefitting` retains public CLI routing. `resources.py` and `maintenance.py`
retain compatibility entry points; deletion policy belongs to
`resource_management`, not those facades. Internal package names do not change
the public commands or existing `data\ALIAS\facefitting` paths.

The Python files directly under `src\ailias` are installed application modules,
not agent scripts. `intake.py` and `workflow.py` implement manual capture and
handoff operations; `settings.py` and `storage.py` own shared configuration and
persistence. Package/CLI entry points and compatibility facades stay importable.
Disposable agent scripts, execution records and diagnostic outputs belong in
the Git-ignored `.agent` directory, not the repository root or installed package.

Storage is local files, not a distributed database or queue. COLMAP's SQLite
database is job scratch. The alias lock excludes cooperating alias writers;
the Unreal project lock separately excludes migrated native writers across aliases.
Unknown external writers are not made safe by either lock.

## Configuration

Precedence is `--data-root` -> selected JSON `data_root` -> default data directory.
`--config FILE` selects the core JSON; relative paths resolve against that file.
Module guides define their own policy selectors, keys and defaults.

| Key in `config\settings.json` | Default | Meaning |
|---|---|---|
| `data_root` | `..\data` | Data directory relative to this configuration |
| `min_photos`, `max_photos` | `8`, `15` | Capture-set count bounds |
| `min_short_side` | `1024` | Minimum image short side, pixels |
| `min_edge_energy` | `18` | Minimum lightweight edge-energy score |
| `max_dark_fraction`, `max_bright_fraction` | `0.6`, `0.6` | Exposure rejection fractions |
| `angle_tolerance` | `10` | Allowed deviation in degrees from required capture views |

Capture gates are separate from face-region imageprocessing thresholds.
Neither establishes anatomical accuracy.

## Execution and ownership

`PipelineStep` names an operation; `ExecutionChain` runs ordered steps and
stops on failure. `StageJournal` records individual stage outcomes in the job
manifest. Domain modules keep their own numerical/native APIs rather than
inheriting a universal pipeline base class.

Resource-owning operations release handles and finalize files in `finally`
blocks. Pure transformations have no artificial cleanup method. A job does
not own the shared Unreal editor and must not stop it on completion.
Nested photo-pipeline jobs release scratch immediately; `defer_retention()`
delays alias-wide major-output retention until the outer pipeline boundary.
Video-producing pipelines release their own scratch but defer whole-alias
history pruning to explicit maintenance. They do not render a separate neutral
preview before producing the requested narrated video.
Image scan/cleansing preserve their separate curation-aware retention behavior.
Their explicit maintenance command delegates image-asset deletion to the same
resource manager, including changed-content, live-reference and link guards.

### Stage measurements

`StageJournal` supports synchronous `execute()` and asynchronous `begin()` /
`finish()` boundaries. Manifest stage entries record UTC timestamps, monotonic
elapsed seconds, outcome and the containing `parent` stage, when any.
Parent durations include their children: never add both to obtain wall time.
The job-level `seconds` includes resource finalization and retention; persistence
and orchestration overhead can exceed the sum of nonoverlapping stages.

Progress observers receive operation/outcome/duration/error-type metadata, not
stored exception messages or causes. Observer failures do not change job
outcomes. Detailed failures remain in the job manifest; persistence failure
during an existing stage failure cannot replace that primary exception.

The shared journal has no eager host-storage dependency. Native reanimation
jobs receive its content-addressed source and verify its hash before loading it
inside Unreal's Python runtime.

| Event | Contract |
|---|---|
| Changed intake or manual handoff | Invalidate downstream manual stages; retain history |
| Missing or changed source | Fail integrity checks instead of rebasing stored hashes |
| Native success without saved files | Fail publication; a receipt alone is insufficient |
| Native host timeout | Do not assume cancellation; preserve files while a writer may remain active |
| Editor failure after writes | Report possible partial packages; do not delete external project assets |
| Hard termination | May leave interrupted jobs; inspect ownership before recovery |

## Resource cleanup

```powershell
python -m ailias cleanup --alias subject
python -m ailias cleanup --alias subject --apply
python -m ailias cleanup --alias subject --scratch-only
python -m ailias cleanup --alias subject --prune-job 'facefitting\appearance\OBSOLETE_REVISION'
```

Default cleanup is a dry run. Apply mode revalidates the plan before removing
exact owned files. Successful producing jobs normally invoke major-output retention;
video-producing jobs defer it to this explicit command to keep completion local.
image scan/cleansing retain their explicit curation-aware maintenance path.

| Resource | Retention rule |
|---|---|
| Major products | Keep newest successful products, active jobs and transitive dependencies |
| Damaged newest product | Preserve the last intact successful product |
| Canonical selections/reviews | Follow verified references when deriving live dependencies |
| Originals, tools and models | Not job scratch; not removed by major-output retention |
| Changed, linked or unregistered files | Preserve rather than infer ownership |
| Job intermediates | Remove unreferenced owned scratch; retain manifests and bounded diagnostics |
| Obsolete manifests | Archive lineage and leave pruning records with surviving references |
| Unreal packages | Inspect dependencies in-editor; remove eligible exact files only after the editor stops |

Native inspection uses the Asset Registry without loading obsolete assets.
A running editor receives a deferred retention plan; cleanup never shuts it
down to reclaim space. Owned-editor exit invokes deferred cleanup after releasing
the session lock. Before offline deletion, the adapter checks the complete
package inventory/hashes, dependency closure, dirty packages, links and other
potential project users. Changed or uncertain state defers deletion.

Retention receipts live under `maintenance\retention`; explicit scratch/whole-job
cleanup uses `maintenance\cleanups`. Before and after native work, retention
revalidates job/canonical metadata, retained roots, live dependencies and hashes.
Cleanup failures are recorded without replacing the original job failure.
Reanimation keeps verified video, not image exports. `--keep-frames` explicitly
selects diagnostic image-sequence output.

## Documentation and media

Usage guides own procedures, inputs, outputs and configuration meanings.
Development guides own implementation contracts and tests. Execution histories,
timing records and investigation notes belong in ignored agent records or job
manifests, not manuals.

README thumbnails are reduced existing products, not new model generation.
[Media provenance](images/pipeline/provenance.json) records source hashes and
display transforms. Portrait thumbnails must fit within 256x256 pixels.
Distinguish format illustrations from matched-camera validation.

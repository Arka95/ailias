[Back to REPO_ROOT](../../README.md) | [Command guide](README.md)

# Photo-to-character workflow internals

## Ownership

`facefitting\cli.py` preserves the public command namespace. Implementation is
split by responsibility rather than by CLI spelling:

| Package | Responsibility |
|---|---|
| `head_reconstruction` | Evidence preparation, numerical fitting, mesh export and Blender appearance |
| `metahuman` | Native Character import/conform, feature edits, assembly and saved-asset checks |
| `pipelines` | Cross-domain photo-to-character and video-refinement execution |
| `imageprocessing` | Shared dense observations, parsing and per-photo feature evidence |
| `generation` | Shared native tools, project-owned editor and independent native processes |
| `resource_management` | Scratch finalization and dependency-safe major-output retention |

See [head reconstruction internals](../head_reconstruction/dev.readme.md) and
[native asset internals](../metahuman/dev.readme.md) for domain algorithms.
Existing `facefitting` data paths remain stable.

## Execution

`PhotoToCharacterPipeline` in `pipelines\photo_to_character.py` builds an
`ExecutionChain` with five named phases:

| Phase | Boundary |
|---|---|
| Prepare evidence and fit identity | Verify prerequisites/sources; compute fresh observations and shared shape |
| Prepare mesh and tracking | Create Blender appearance, export the actual native input and tracking view |
| Create and verify Character | Import/conform a new native Character and verify saved geometry |
| Apply native appearance | Apply a validated native or photo-colour appearance policy |
| Render and validate preview | Produce a new editor-native image and check retained outputs |

`StageJournal` persists stage outcomes and sends diagnostic events to stderr.
Failure stops subsequent phases. The public function wrapper preserves the
call signature; it delegates execution and finalization to the owning object.
The chain is local and synchronous, not a background queue or resume cache.

`video-create --evidence` consumes retained images/timestamps, not the original
video decoder. It computes fresh landmarks/parsing and does not reuse an old
head, identity fit or native response. `--photos-run` adds hash-verified accepted
originals across the selected scan's groups; detection crops do not replace
source-coordinate geometry.

The Blender stage is mandatory for this command. Missing scenes, failed renders
or altered export geometry stop execution; there is no silent direct-OBJ fallback.
Runtime assembly and narrated rendering are separate operations.

## Fitting restrictions and search

One shared identity is fitted with per-view camera and expression parameters.
The neutral export excludes those expressions. Landmark predictions and the
head-model prior are not independent measurements of hidden anatomy.

The standard policy compares configured identity capacities. The optional
`--fit-search-config` evaluates the explicitly selected capacity/prior/bound
combinations and records failed candidates as well as successful ones.
It is not a mandatory step for appearance edits or rendering.

Within one requested search, compatible zero-identity reference fits are reused
by expression model/prior/bound. Failed references are retried; reuse does not
remove candidates, images, held-out scoring or the final all-view fit.
`view_batch_size` limits Jacobian memory, not the number of contributing views.

Held-out evaluation fixes identity but still fits camera/expression on each
validation image. It measures conditional projected agreement, not independent
3D accuracy. The final all-view fit is not a held-out score.

## Fitting architecture

Numerical model loading, camera initialization, CPU/CUDA optimization and
geometry serialization belong to
[head reconstruction](../head_reconstruction/dev.readme.md#fitting).
Cross-domain rendering/appearance policies belong to `pipelines\settings.py`;
they are not numerical fitting settings.

## Appearance and native boundaries

Photo-colour projection creates an atlas in the fitted mesh's UV coordinates.
A native Character has different topology/UVs and native skin/groom controls.
Conforming geometry does not transfer its Blender material, hair or iris.

Native appearance mapping constructs explicit feature profiles; the
[MetaHuman adapter](../metahuman/dev.readme.md#appearance-and-assembly)
owns applying, saving and independently checking those properties.
Unmeasurable beard/lash evidence and stock groom approximations remain explicit.
Appearance-only jobs must preserve geometry and source-package identities.

## Coordinate and persistence contracts

| Boundary | Convention or invariant |
|---|---|
| OBJ export | `(GNM Z, X, Y) * 100` |
| Imported static preview | `(GNM Z, -X, Y) * 100` |
| Native target/camera | `(GNM X, Z, Y) * 100` |
| Photo contour | Full source-frame pixels and retained perspective camera; no cropped principal-point substitution |
| Optional baseline | Verify pinned dependencies; never refit or overwrite the selected scene |
| Native save | Commit state, save package, verify actual file/hash |
| Assembly | Compiled actor/rig and independent reload are separate from API return values |
| Native edits | Duplicate the Character; retain original source hashes |

Coordinate scaling does not establish physical calibration. Native reload
agreement establishes persistence, not photo-to-native resemblance.

## Resource lifecycle

The outer pipeline owns its job directory and calls `cleanup()` in `finally`.
It uses `ResourceManager.defer_retention()` while child jobs run: each releases
scratch, while alias-wide retention runs once at the outer boundary.
Native adapters own request-scoped resources, not the project-owned editor.
Unconfirmed native completion defers file finalization.
See [cleanup safeguards](../dev.readme.md#resource-cleanup).

## Configuration

The [command tables](README.md#configuration) define public selectors and
defaults. `head_reconstruction\settings.py` owns numerical/evidence policies;
`pipelines\settings.py` owns cross-domain appearance/review policies.
Workers are copied by content hash into jobs so later source edits do not
rewrite the implementation recorded for an existing output.

## Evidence

Manifests record source identities, policy, observations, candidate outcomes,
artifacts and stage receipts. Inspect actual saved assets and pixels in addition
to numerical summaries. No manifest success state certifies likeness.
Execution measurements and run histories are not part of this manual.

## Validation and development

```powershell
python -m unittest discover -s src\ailias\head_reconstruction -t src -q
python -m unittest discover -s src\ailias\metahuman -t src -q
python -m unittest discover -s src\ailias\pipelines -t src -q
python -m unittest ailias.tests.test_resources ailias.tests.test_retention -q
```

Units use synthetic geometry, fake observations/native editors and temporary
files. They cover source integrity, fitting/search behavior, stage failures,
geometry handoffs, native persistence contracts and cleanup.
Explicit CUDA checks are separate:

```powershell
$env:AILIAS_CUDA_FITTING_TESTS = "1"
python -m unittest discover -s src\integration_tests\facefitting -p test_cuda.py -v
python -m unittest discover -s src\integration_tests\facefitting -p test_appearance_cuda.py -v
```

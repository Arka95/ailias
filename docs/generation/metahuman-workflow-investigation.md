[Back to REPO_ROOT](../../README.md) | [Native asset guide](../metahuman/README.md)

# Photo references to MetaHuman: routes and gates

## 1. Problem Statement

A fitted head, an editable Character and an assembled runtime actor are different
representations. Dense projected landmarks do not independently measure hidden
anatomy; successful conform/rigging does not establish likeness.
Validate the photograph-to-mesh and mesh-to-native boundaries separately.

## 2. Implementation

### Route selection

| Route | Useful property | Constraint |
|---|---|---|
| Shared head prior fitted to photos | Coherent shape with per-view camera/expression | Prior-supported hidden anatomy; not a calibrated scan |
| Consistent photogrammetry -> mesh | Multi-view surface observations | Needs rigid subject/overlap; fragments are not a complete head |
| Arbitrary mesh -> From Custom Mesh | Head-only native solve accepts non-native topology [1] | Check actual shape drift, camera/axes and saved state |
| Mesh -> solved Identity -> Character | Alternative tracked-frame handoff [2] | Identity promoted frames are not the numerical fitter's view set |
| Compatible template -> From Template | Native topology/UV-constrained handoff | Arbitrary GNM topology is not established compatibility |
| Commercial photo fitting | Authored geometry and explicit native texture transfer [3] | Separate tool/runtime/licensing evaluation |
| Learned identity comparator | Independent predicted geometry, such as MICA [4] | Model terms and native conversion remain separate |

Use the existing head-only Custom Mesh route for arbitrary fitted geometry.
Do not claim it is the best likeness method without comparable measurements.
Identity is an alternative, not a mandatory extra solve after every conform.

### Boundaries and persistence

`head_reconstruction` owns observations, shared shape and export.
`metahuman` owns static import, native conform and Character edits.
`pipelines` composes them; `generation` supplies process/session adapters.
Static import alone creates neither an editable Character nor a rig.

Native targets/cameras and static preview use different declared coordinate
mappings. Core Data readiness is reported separately from mesh import.
Cloud rig/texture requests require explicit selection.
Commit native state and verify actual saved packages in an independent process.
See [coordinate contracts](../facefitting/dev.readme.md#coordinate-and-persistence-contracts).

Geometry, skin/iris, scalp/facial hair and runtime assembly need distinct checks.
Do not treat hair silhouette as bare skull or native picker values as calibrated
skin colour. Animation must inspect blink, lip seal, smile/jaw and deformation.

### Capacity and backpressure

A planning set of 64 images with 478 predictions per image contains
`64 * 478 = 30,592` predicted points, not independent 3D measurements.
Jacobian batching controls memory without discarding views.
Alias and project locks bound cooperating writers; native compile/render and
the single local GPU/disk remain bottlenecks and failure boundaries.
No distributed queue or replicated storage is introduced.

## 3. Success Criteria

| Gate | Required evidence |
|---|---|
| Equivalent fitting | Same sources, model, split and objective before optimization comparisons |
| Geometry | Matched-view regional checks and explicit unobserved anatomy |
| Native handoff | Correct axes/camera, committed saved state and measured source-to-native drift |
| Runtime | Compatible compiled actor/rig and appropriate groom/LOD checks |
| Appearance | Native-UV correctness, observed coverage and matched-light review |
| Final likeness | Actual visual acceptance; no single score, API return or movie substitutes |

## 4. Task Breakdown

| Area | Acceptance procedure |
|---|---|
| Source fitting | Synthetic contracts plus explicitly requested real evidence evaluation |
| Native handoff | Inspect source geometry, conform result and independent reload separately |
| Regional diagnostics | Compare front/oblique/profile under a common camera/expression policy |
| Appearance/runtime | Save/reload properties, inspect actual rendered skin/grooms and deformation |

Detailed functional contracts:
[head reconstruction](../head_reconstruction/dev.readme.md),
[MetaHuman](../metahuman/dev.readme.md) and
[quality criteria](../todo/README.md).

## 5. Reference Links

1. [Epic: From Custom Mesh](https://dev.epicgames.com/documentation/metahuman/metahuman-creator-from-custom-mesh-tool-in-unreal-engine?application_version=5.8).
2. [Epic: From Mesh](https://dev.epicgames.com/documentation/metahuman/from-mesh?application_version=5.8).
3. [KeenTools photo-to-MetaHuman](https://keentools.io/integrations/fbb-mh).
4. [MICA implementation and terms](https://github.com/Zielon/MICA).
5. [Epic: assembly](https://dev.epicgames.com/documentation/metahuman/assembly?application_version=5.8).
6. [Epic: Core Data installation](https://dev.epicgames.com/documentation/metahuman/getting-started-with-metahuman-creator-in-unreal-engine?application_version=5.8).

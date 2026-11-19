[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Head reconstruction

Estimate one head shape from multiple photographs while allowing each view
its own camera and expression. Export a neutral mesh and prepare Blender
geometry/appearance products. Native Unreal assets belong to
[MetaHuman](../metahuman/README.md).

## Inputs and outputs

| Input | Output |
|---|---|
| Curated image group and current cleansing | Dense observations, fitted parameters/cameras, neutral OBJ and clay review scene |
| Retained video-frame manifest | Sampled-image evidence for the cross-module photo-to-character workflow |
| Approved mesh and matching review | Photo-colour atlas, coverage, Blender scene and hair-free mesh handoff |

Commands keep the public `facefitting` spelling:

```powershell
python -m ailias facefitting models subject
python -m ailias facefitting fit subject RUN_ID face-001
python -m ailias facefitting appearance subject FIT_JOB
```

`appearance` requires a matching `neutral_clay_geometry` approval record.
See the [command and configuration reference](../facefitting/README.md) for
review requirements, video inputs and policy selectors.

## Interpretation

GNM is the statistical head model used as a shape prior. Photo landmarks are
model predictions, not a calibrated scan. Hidden anatomy and absolute scale
remain uncertain. Photo colour includes illumination; missing atlas regions
are filled, not recovered. Procedural hair is an approximation, not a bound
native groom or strand reconstruction.

Preparing models is explicit. CPU/CUDA selection follows configuration without
silently switching devices or replacing shared runtime dependencies.

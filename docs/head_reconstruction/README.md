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
| Retained `FaceFeatures` per view + explicitly supplied head model/policy | Neutral vertices/topology, fitted parameters/cameras and diagnostics, without repeating image inference |
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

## Reconstruct from retained FaceFeatures

The Python API consumes one known person's features, keyed by unique view IDs
(not person/identity IDs). Imageprocessing stays independent of this module.
Use the existing pinned GNM head and MediaPipe correspondence assets; loading
them is explicit, and the fitter never loads MediaPipe or original images.

```python
from pathlib import Path
from ailias.head_reconstruction.geometry.feature_reconstruction import (
    HeadReconstructionError, reconstruct_head_from_features,
)
from ailias.head_reconstruction.geometry.head_model import HeadModel
from ailias.head_reconstruction.settings import load_policy

policy = load_policy()
model = HeadModel(
    Path(r"data\arka\facefitting\models\gnm-v3.0\gnm_head.npz"),
    Path(r"data\arka\facefitting\models\gnm-mp468-v2\head_dense_468.txt"),
    policy.expression_modes,
)
features_by_view = {
    f"view-{index:06d}": item.extraction.features
    for index, item in enumerate(grouped.selected)
}
try:
    reconstruction = reconstruct_head_from_features(model, features_by_view, policy)
except HeadReconstructionError as error:
    print(error.report)  # Includes per-view exclusions even if fitting cannot start.
    raise

vertices = reconstruction["vertices"]
triangles = reconstruction["triangles"]
parameters = reconstruction["fit"]
```

`grouped` comes from `group_face_images()` in the
[imageprocessing API](../imageprocessing/README.md#simple-python-api).
Saved feature JSON can be restored using `FaceFeatures.from_dict(data)`.
Each record must retain `source_size=[width,height]`, 478 finite source-pixel
points and 478 boolean `on_head` flags. The fitter uses the first 468 points;
outer-eye spacing is derived, not a physical interpupillary measurement.
No live pixels, BiSeNet inference, identity matching, MediaPipe transform or
MediaPipe expression scores are required by this fitting connection.

At least five geometrically usable views are required. Invalid evidence, camera
initialization failure and insufficient template/foreground support are reported
per view. Blur/exposure/expression warnings and large orientation angles do not
reject views in this new path. Camera initialization and template-normal
self-occlusion can still make a particular view numerically unusable.
Existing held-out identity-improvement/frontal-preservation gates and mesh
collapse/flip/stretch checks remain required; failure does not publish a
generic head as a successful personal reconstruction.

The result is in memory: no previews, sidecars, meshes, models or native assets
are saved automatically. Use `geometry.artifact_io.export_obj()` and `save_fit()`
explicitly to export the returned geometry/fit. This is a new Python entry point;
the existing `facefitting fit` CLI, its legacy observation path and pose rules
are unchanged. Pixel-based appearance fitting remains a separate downstream task.

## Interpretation

GNM is the statistical head model used as a shape prior. Photo landmarks are
model predictions, not a calibrated scan. Hidden anatomy and absolute scale
remain uncertain. Photo colour includes illumination; missing atlas regions
are filled, not recovered. Procedural hair is an approximation, not a bound
native groom or strand reconstruction.

Preparing models is explicit. CPU/CUDA selection follows configuration without
silently switching devices or replacing shared runtime dependencies.

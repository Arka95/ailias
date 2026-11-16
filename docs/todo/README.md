[Back to REPO_ROOT](../../README.md)

# Remaining implementation

All items below are open; none is implied complete by a manual handoff receipt.

| Order | Work | Evidence needed before closing |
|---|---|---|
| 1 | Prove source plan's manual milestone | Real photo set, reconstructed/cleaned head, MetaHuman and three-view human baseline |
| 2 | Benchmark reconstruction candidates | Experimental native COLMAP benchmark implemented; 31 real portraits fail overlap gates; select a viable human-head backend and measure accuracy/runtime/VRAM |
| 3 | Extend photo analysis | Face detection, five points, identity grouping and visible-body extraction implemented; pose/expression/occlusion inference and real threshold calibration still open |
| 4 | Prove reconstruction adapter on a real head | Full pipeline and explicit permissive PoC implemented; real photos now produce inspectable textured fragments, but a coherent head remains unproven |
| 5 | Finish Blender and Unreal integration | Blender CUDA cleanup/export/renders verified; UE 5.8 guarded adapter implemented but engine absent; real Identity solve/conform, assembly and editor-lifetime checks still open |
| 6 | Complete multi-signal validation | Aligned 2D/3D landmark and sampled-surface primitives implemented; true mesh-surface, silhouette, matched-camera and calibrated regional checks still open |
| 7 | Confidence-driven recapture | Region evidence, proposed yaw/pitch/distance and diminishing-value stopping rule |
| 8 | Bounded refinement and rollback | Measurable multi-view acceptance, preserved revisions, one-step rollback and iteration cap |
| 9 | Animation validation and sign-off | Blink, brows, smile, mouth, purse, jaw and speech-like expression results; human approval |
| 10 | Session curation and recovery UX | Explicit photo removal, history inspection and rollback commands without data loss |

Completed infrastructure: CUDA imageprocessing on all 192 requested photos,
corrected grouping, one shared workspace environment, source references instead
of copies, reusable generated assets, cleanup of superseded runs, and explicit
human-curated exemplar filtering with uncertain-crop review.
Native tools are installed once in the workspace; COLMAP-to-Blender GPU
integration passes on a synthetic object, with transient resources cleaned.
See the [imageprocessing validation](../imageprocessing/research.md) and
[generation evidence](../generation/dev.readme.md#validation).
Automatic identity accuracy and reconstruction suitability remain unverified;
55 candidate groups do not establish 55 people.

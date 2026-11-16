[Back to REPO_ROOT](../../README.md)

# Remaining implementation

All items below are open; none is implied complete by a manual handoff receipt.

| Order | Work | Evidence needed before closing |
|---|---|---|
| 1 | Prove source plan's manual milestone | Real photo set, reconstructed/cleaned head, MetaHuman and three-view human baseline |
| 2 | Benchmark reconstruction candidates | Accuracy/runtime/VRAM measurements on target machine; select backend |
| 3 | Extend photo analysis | Face detection, five points, identity grouping and visible-body extraction implemented; pose/expression/occlusion inference and real threshold calibration still open |
| 4 | Add selected reconstruction adapter | Explicit opt-in; recovered cameras, textures and reproducible settings; failure/retry tests |
| 5 | Automate Blender and Unreal | Verified installed-version APIs; approved local operations; actual tool integration checks |
| 6 | Complete multi-signal validation | Aligned 2D/3D landmark and sampled-surface primitives implemented; true mesh-surface, silhouette, matched-camera and calibrated regional checks still open |
| 7 | Confidence-driven recapture | Region evidence, proposed yaw/pitch/distance and diminishing-value stopping rule |
| 8 | Bounded refinement and rollback | Measurable multi-view acceptance, preserved revisions, one-step rollback and iteration cap |
| 9 | Animation validation and sign-off | Blink, brows, smile, mouth, purse, jaw and speech-like expression results; human approval |
| 10 | Session curation and recovery UX | Explicit photo removal, history inspection and rollback commands without data loss |

Completed infrastructure: CUDA imageprocessing on all 192 requested photos,
corrected grouping, one shared workspace environment, source references instead
of copies, reusable generated assets and cleanup of superseded runs.
See the [validation record](../imageprocessing/research.md).
Automatic identity accuracy and reconstruction suitability remain unverified;
55 candidate groups do not establish 55 people.

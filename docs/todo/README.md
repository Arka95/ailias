[Back to REPO_ROOT](../../README.md)

# Remaining implementation

All items below are open; none is implied complete by a manual handoff receipt.

| Order | Work | Evidence needed before closing |
|---|---|---|
| 1 | Prove source plan's manual milestone | Real photo set, reconstructed/cleaned head, MetaHuman and three-view human baseline |
| 2 | Compare head-generation quality | GNM baseline now produces a coherent head; fresh CUDA pipeline 42.187s versus 372.859s CPU with better held-out error. Independent silhouette/depth/likeness and competing-backend comparison remain open |
| 3 | Calibrate cleansing and extend photo analysis | Semantic head-only alpha, bounded colour correction, reversible roll/position/scale alignment and relative-light ranking implemented. Generic detector retries recover close-ups; all ready heads receive paginated previews. Improve review/edge/colour calibration; five-point view bins are approximate, not dense geometry; expression/occlusion assessment remains open |
| 4 | Validate fitted head against independent evidence | Coherent GNM neutral head, mesh checks, dense overlays and corrected review renders exist; prior-driven anatomy, texture and likeness still need review |
| 5 | Finish Blender and Unreal integration | Blender CUDA verified; UE 5.8.4 owned-project commandlet verified Python conforming APIs. Install missing MetaHuman Core Data, validate actual head-only Custom Mesh, compare shape drift, then assembly and editor lifetime; [research](../generation/metahuman-workflow-investigation.md) |
| 6 | Complete multi-signal validation | Aligned 2D/3D landmark and sampled-surface primitives implemented; true mesh-surface, silhouette, matched-camera and calibrated regional checks still open |
| 7 | Confidence-driven recapture | Region evidence, proposed yaw/pitch/distance and diminishing-value stopping rule |
| 8 | Bounded refinement and rollback | Measurable multi-view acceptance, preserved revisions, one-step rollback and iteration cap |
| 9 | Animation validation and sign-off | Blink, brows, smile, mouth, purse, jaw and speech-like expression results; human approval |
| 10 | Session curation and recovery UX | Explicit photo removal, history inspection and rollback commands without data loss |
| 11 | Validate the separate [reanimation module](../reanimation/README.md) on an existing assembled MetaHuman | Real PocketTTS/testuser narration and 742-frame native speech solve/reload succeeded; NVIDIA encoding passed. Corrected GPU fixture writes green frames but both editors crash on shutdown (`0xC0000005`); do not ignore the exit code. Also need genuine rigged Blueprint, actual face/head render and visual review. Does not authorize restarting stopped likeness repair |

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
See the [plain-language cleansing decisions](../imageprocessing/dev.readme.md#research-decisions-in-plain-language)
and [fitting/appearance decisions](../generation/dev.readme.md#research-decisions-in-plain-language).

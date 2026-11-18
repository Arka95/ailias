[Back to REPO_ROOT](../../README.md) | [Decisions](../analysis.md)

# Implementation backlog

Open engineering work, ordered by dependency and product impact.
Technical completion and visual quality are separate acceptance conditions.

| Priority | Work | Existing foundation | Completion evidence |
|---|---|---|---|
| P0 | Native facial-feature refinement | Preserved Blender geometry, reference photos and conformed Character | Code-driven regional fitting, matched-camera before/after metrics and visible resemblance improvement |
| P0 | Native appearance transfer | Photo-colour atlas, UV geometry and default-textured runtime actor | Correct native UV/material bake, explicit skin-tone treatment and saved/reloaded comparison |
| P0 | Hair, brows, moustache and beard | Photo evidence and approximate unbound Blender scalp groom | Automated compatible selection/binding, correct root alignment and deformation/LOD checks |
| P1 | Persistent native rig/build/render orchestration | Existing rig, source textures, session receipts and [exact draft edit plan](../reanimation/pipeline-optimization-investigation.md#decided-code-changes) | Same owned working editor across warm operations, project-wide mutation exclusion, correct async completion and one final independent saved-asset check; no default-character substitution |
| P1 | Dependency-aware iteration and measured fitting optimization | Existing stage manifests, explicit native/reel policies and CUDA search | No rig/refit/skin rebuild for lighting-only changes; opt-in compatible Performance reuse; zero-mode reference reuse preserves all 162 candidates; measured cold/warm latency and RAM/VRAM evidence |
| P1 | Matched-view multi-signal validation | Landmark/sampled-surface metrics and visual comparison | Automated camera-consistent overlays, regional/silhouette checks and bounded accept/reject criteria |
| P1 | Refinement/rollback loop | Immutable revisions, source hashes and integrity checks | Iteration cap, measurable nonregression and recoverable prior candidate |
| P2 | Calibrate mask/colour confidence | Semantic head alpha, bounded harmonization and relative-light evidence | Edge/colour reference fixtures, uncertainty reporting and reliable abstention |
| P2 | Anatomical dimensions | Supplied corresponding geometry comparisons | Defined anatomical endpoints, calibrated scale and independent measurement validation |
| P2 | Resume audio-only animation | Saved Performance and rig rendering | Render an existing verified Performance without repeating the solve |
| P2 | Expression/performance validation | Full audio-driven render and technical media gates | Blink, lip seal, brows, jaw/smile and speech-motion inspection |
| P3 | Optional apparel | Existing compatible character wardrobe | Independent garment asset/style/colour controls, fit/deformation tests and backdrop compatibility |
| P3 | Capture/recovery UX | Source history, curation, manual handoffs and baselines | Region-driven recapture guidance, explicit recovery/rollback and no data loss |

Core Data, native Character conform, face rigging, runtime assembly and real
video encoding already exist locally; reinstalling/repeating those is not a
substitute for the missing resemblance/appearance work.

Current code/quality boundaries: [facefitting](../facefitting/README.md),
[reanimation](../reanimation/README.md) and the [root comparison](../../README.md#current-state---below-expectations).

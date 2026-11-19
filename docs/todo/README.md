[Back to REPO_ROOT](../../README.md) | [Design decisions](../analysis.md)

# Quality and extension criteria

Requirements below define acceptance for visual-quality work and optional
extensions. Saved-asset validity and visual acceptance are separate gates.

| Priority | Area | Required acceptance evidence |
|---|---|---|
| P0 | Facial geometry, including nose | Matched-camera regional measurements, visible resemblance improvement and no regression of protected features |
| P0 | Native complexion | Saved/reloaded comparison under matched lighting; no synthetic photo fill used to conceal material or colour errors |
| P0 | Brows and eyelashes | Visible placement, density and colour checks in neutral and animated views; requested native settings persist |
| P0 | Hair, moustache and beard | Compatible selection/binding, correct root alignment and deformation/LOD checks |
| P1 | Multi-signal validation | Camera-consistent overlays, regional/silhouette checks and bounded accept/reject criteria |
| P1 | Refinement and rollback | Explicit iteration cap, measurable nonregression and recoverable prior candidate |
| P1 | Remaining native lifecycle migration | Preserve project exclusion, request ownership, asynchronous completion and independent publication checks |
| P1 | Fitting optimization | Equivalent candidate/evidence evaluation; profile real bottlenecks without reducing correctness gates |
| P2 | Mask and colour confidence | Reference fixtures, uncertainty reporting and reliable abstention |
| P2 | Anatomical dimensions | Defined endpoints, calibrated scale and independent measurement validation |
| P2 | Audio-performance reuse | Explicitly render an existing verified Performance without silently reusing or repeating a solve |
| P2 | Expression quality | Inspect blink, lip seal, brows, jaw/smile and speech-motion alignment |
| P3 | Apparel controls | Independent garment asset/style/colour selection, fit/deformation and backdrop checks |
| P3 | Capture and recovery | Region-driven recapture guidance and explicit recovery without data loss |

Implementation contracts: [head reconstruction](../head_reconstruction/README.md),
[MetaHuman](../metahuman/README.md) and [reanimation](../reanimation/README.md).

[Back to REPO_ROOT](../README.md)

# Scope and decisions

Source: [AI MetaHuman Creation - Action Plan](https://microsoftapc-my.sharepoint.com/:w:/g/personal/bhowmikarka_microsoft_com/cQrnuehcebS3SKX4dIPNvRcREgUD__30rs8H5Grb9-Fy03BGMA),
read through WorkIQ on November 16, 2026. Project/folder name `ailias` was supplied
by the owner. Mode: AUTO + RAPID_DEVELOPMENT.

The source explicitly requires a mostly manual end-to-end run before automation.
This implementation provides preparation/recording tools, plus local image
processing and geometric comparison primitives. **The manual milestone itself
has not been achieved**: a real source-photo directory has been analyzed, but no
controlled capture set, reconstruction benchmark, cleaned head, Unreal conversion
or final likeness review has been verified.

| Decision | Chosen option/configuration | Tradeoff / alternatives | Acceptance evidence |
|---|---|---|---|
| Initial interface | Local Python CLI | Small and testable; no live capture UI | Commands and persistence tested with generated images |
| Persistence | Per-session JSON, hashed external sources and canonical generated assets | Inspectable/resumable; originals must remain available, no multi-machine concurrency | Atomic save, writer exclusion, integrity and invalidation tests |
| Intake quality gate | Pillow resolution/edge/exposure heuristics; optional face-specific preprocessing | Whole-image checks can miss facial blur; face thresholds still need calibration | Deterministic units and separate local folder integration |
| Pose/expression | Human-reviewed values | Face detection/five points are available, but no automatic pose/expression inference | Explicit capture flags and targeted missing-view output |
| Imageprocessing | Explicit CPU or reused CUDA OpenCV YuNet/SFace/PP-HumanSeg behind a Python protocol | Local and reviewable; identity grouping can over-split and body masks are coarse | [Model research and validation](imageprocessing/research.md) |
| Shared environment | One `C:\projectWorkspace\.venv` junction to the existing working Thea environment | No duplicate venv/CUDA packages; missing lightweight dependencies are shared and target must remain in place | ailias imports, CUDA targets and unit tests verified; app-local overlay removed |
| Job cleanup | Job-local finalizers plus explicit verified legacy migration/protected-run pruning | No original-photo copies; reuse generated assets, protect selected/history-dependent results; hard kills need maintenance | Units cover failures, asset reuse, missing originals, migration and protected runs |
| Geometry errors | No-reflection SVD landmark alignment, optional sampled 3D surface distances | No false single likeness score; not full point-to-triangle mesh evaluation | Known synthetic geometry and transform/error unit tests |
| Reconstruction | Manual receipt, no selected backend | Avoid speculative GPU stack; no automatic mesh generation | Full per-photo camera record contract; geometry still manually verified |
| Blender/Unreal | Manual receipts | No unverifiable editor API automation | Required artifacts and ordered gates, not external-tool execution |
| Baseline | Three-view contact sheet plus human note | Supports initial visual comparison; no numeric likeness score | Real image output from synthetic fixtures |

The source's future pipeline additionally calls for regional confidence based on
visibility, triangulation baseline and noise; multi-signal landmark/silhouette/
surface evaluation; active recapture; bounded reversible 3-8 iteration refinement;
expression validation and human sign-off. The new landmark/sampled-surface metric
primitives are not a completed automated MetaHuman validation loop.

## Verified external references

- [Epic: Mesh to MetaHuman](https://dev.epicgames.com/documentation/metahuman/mesh-to-metahuman?application_version=5.8)
- [Epic: MetaHuman Creator Python scripting](https://dev.epicgames.com/documentation/metahuman/python-scripting-for-metahuman-creator)
- [Epic: MetaHuman 5.8 release notes](https://dev.epicgames.com/documentation/metahuman/metahuman-5-8-release-notes-in-unreal-engine)

Official pages were checked on November 16, 2026; Unreal/Blender have not been
probed. GPU imageprocessing has been verified separately; see its research log.
Version availability in documentation is not proof of
local installation or compatibility.

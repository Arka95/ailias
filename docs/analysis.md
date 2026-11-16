[Back to REPO_ROOT](../README.md)

# Scope and decisions

Source: [AI MetaHuman Creation - Action Plan](https://microsoftapc-my.sharepoint.com/:w:/g/personal/bhowmikarka_microsoft_com/cQrnuehcebS3SKX4dIPNvRcREgUD__30rs8H5Grb9-Fy03BGMA),
read through WorkIQ on November 16, 2026. Project/folder name `ailias` was supplied
by the owner. Mode: AUTO + RAPID_DEVELOPMENT.

The source explicitly requires a mostly manual end-to-end run before automation.
This implementation provides preparation/recording tools, plus local image
processing, geometric comparison and experimental native generation. **The manual
milestone itself has not been achieved**: real portraits were benchmarked, but no
controlled capture set, reconstructed human head, Unreal conversion or final
likeness review has been verified.

| Decision | Chosen option/configuration | Tradeoff / alternatives | Acceptance evidence |
|---|---|---|---|
| Initial interface | Local Python CLI | Small and testable; no live capture UI | Commands and persistence tested with generated images |
| Persistence | Per-session JSON, hashed external sources and canonical generated assets | Inspectable/resumable; originals must remain available, no multi-machine concurrency | Atomic save, writer exclusion, integrity and invalidation tests |
| Intake quality gate | Pillow resolution/edge/exposure heuristics; optional face-specific preprocessing | Whole-image checks can miss facial blur; face thresholds still need calibration | Deterministic units and separate local folder integration |
| Pose/expression | Human-reviewed values | Face detection/five points are available, but no automatic pose/expression inference | Explicit capture flags and targeted missing-view output |
| Imageprocessing | Explicit CPU or reused CUDA OpenCV YuNet/SFace/PP-HumanSeg behind a Python protocol | Local and reviewable; identity grouping can over-split and body masks are coarse | [Model research and validation](imageprocessing/research.md) |
| Curated portrait rejection | Explicit human keep/delete labels, SFace exemplar contrast, exact-crop memory and separate uncertain review | No new model stack; similarity does not certify crop quality, so borderline examples are not deleted | Source-held-out diagnostic and pipeline tests; [curation research](imageprocessing/research.md#curated-portrait-filtering) |
| Shared environment | One `C:\projectWorkspace\.venv` junction to the existing working Thea environment | No duplicate venv/CUDA packages; missing lightweight dependencies are shared and target must remain in place | ailias imports, CUDA targets and unit tests verified; app-local overlay removed |
| Job cleanup | Job-local finalizers plus explicit verified legacy migration/protected-run pruning | No original-photo copies; reuse generated assets, protect selected/history-dependent results; hard kills need maintenance | Units cover failures, asset reuse, missing originals, migration and protected runs |
| Geometry errors | No-reflection SVD landmark alignment, optional sampled 3D surface distances | No false single likeness score; not full point-to-triangle mesh evaluation | Known synthetic geometry and transform/error unit tests |
| Reconstruction | Experimental COLMAP 4.2.1 CUDA, with view-graph/SfM/dense gates; manual workflow kept separate | Native shared tool, no new model stack; generic photogrammetry is not a validated face-prior backend | Full synthetic native pipeline passed; 31 real eligible portraits blocked for insufficient overlap; [evidence](generation/dev.readme.md#validation) |
| Blender | Shared Blender 4.5.15 LTS, minimal cleanup, preserved UVs/cameras and CUDA renders | Avoid invented anatomy, smoothing and automatic component deletion; holes/lighting/likeness still need review | Actual textured mesh export and three non-flat renders on RTX 4070 SUPER |
| Unreal | Guarded UE 5.8 import/conform adapter, explicit cloud opt-in; manual Identity solve | Installed engine/plugins/project required; API fakes are not engine validation; no assembly yet | Missing engine correctly blocks without placeholders; local engine integration remains unverified |
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

Official pages and native prerequisites were checked on November 16, 2026.
Blender/COLMAP ran locally; no Unreal installation was found. See
[generation evidence and primary references](generation/dev.readme.md) and the
imageprocessing research log. Documentation availability is not proof of local
engine installation or compatibility.

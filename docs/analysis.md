[Back to REPO_ROOT](../README.md)

# Scope and decisions

Source: [AI MetaHuman Creation - Action Plan](https://microsoftapc-my.sharepoint.com/:w:/g/personal/bhowmikarka_microsoft_com/cQrnuehcebS3SKX4dIPNvRcREgUD__30rs8H5Grb9-Fy03BGMA),
read through WorkIQ on November 16, 2026. Project/folder name `ailias` was supplied
by the owner. Mode: AUTO + RAPID_DEVELOPMENT.

The source explicitly requires a mostly manual end-to-end run before automation.
This implementation provides preparation/recording tools, plus local image
processing, geometric comparison and experimental native generation. **The manual
milestone itself has not been achieved**: real portraits were benchmarked, but no
controlled capture set, Unreal conversion or final likeness review has been
verified. A coherent photo-conditioned GNM head now exists, with
[measured GPU speed/held-out evidence](facefitting/dev.readme.md#evidence).

| Decision | Chosen option/configuration | Tradeoff / alternatives | Acceptance evidence |
|---|---|---|---|
| Initial interface | Local Python CLI | Small and testable; no live capture UI | Commands and persistence tested with generated images |
| Persistence | Per-session JSON, hashed external sources and canonical generated assets | Inspectable/resumable; originals must remain available, no multi-machine concurrency | Atomic save, writer exclusion, integrity and invalidation tests |
| Intake quality gate | Pillow resolution/edge/exposure heuristics; optional face-specific preprocessing | Whole-image checks can miss facial blur; face thresholds still need calibration | Deterministic units and separate local folder integration |
| Pose/expression | Human-reviewed capture values; separate automatic five-point pose and dense MediaPipe/expression observations | Automatic predictions do not become human capture attestations | Explicit capture flags; dense held-out head-fitting evidence |
| Imageprocessing | Explicit CPU or reused CUDA OpenCV YuNet/SFace/PP-HumanSeg behind a Python protocol | Local and reviewable; identity grouping can over-split and body masks are coarse | [Model research and validation](imageprocessing/research.md) |
| Curated portrait rejection | Explicit human keep/delete labels, SFace exemplar contrast, exact-crop memory and separate uncertain review | No new model stack; similarity does not certify crop quality, so borderline examples are not deleted | Source-held-out diagnostic and pipeline tests; [curation research](imageprocessing/research.md#curated-portrait-filtering) |
| Head-only cleansing | Face-anchored BiSeNet labels plus MODNet opacity; bounded scene-neutral white balance and subject-relative exposure; separate immutable assets | Person matting alone retains shirts. Semantic estimates can fail: flag lower-boundary leakage/poor face coverage for review; unsupported colour correction is skipped | Functional head/colour/native-mask tests and real CUDA photo validation; [plain-language decisions](imageprocessing/dev.readme.md#research-decisions-in-plain-language) |
| Stable head analysis and frontal recall | Generic no-face-only detector scale retries; reversible eye-axis/position/scale alignment; visible-skin relative-light ranking, full paginated previews | Preserve unwarped native geometry; do not synthesize frontal views or infer calibrated skin tone. Pose proxies and lighting heuristics can still fail | All 192 originals rescanned; all three nominated fronts selected without filename rules; 166 aligned heads, 664 verified products, source-coordinate inverse checks; [validation](imageprocessing/dev.readme.md#validation) |
| Shared environment | One `C:\projectWorkspace\.venv` junction to the existing working Thea environment | No duplicate venv/CUDA packages; missing lightweight dependencies are shared and target must remain in place | ailias imports, CUDA targets and unit tests verified; app-local overlay removed |
| Job cleanup | Job-local finalizers plus explicit verified legacy migration/protected-run pruning | No original-photo copies; reuse generated assets, protect selected/history-dependent results; hard kills need maintenance | Units cover failures, asset reuse, missing originals, migration and protected runs |
| Geometry errors | No-reflection SVD landmark alignment, optional sampled 3D surface distances | No false single likeness score; not full point-to-triangle mesh evaluation | Known synthetic geometry and transform/error unit tests |
| Reconstruction | Experimental COLMAP 4.2.1 CUDA; strict default and explicit permissive PoC profile, separate from manual workflow | Native shared tool, no new model stack; early visual output may be fragmented rather than blocked | Strict 31-photo run blocked; permissive initialization/dense consistency produced 6,164 textured faces in 13 disconnected pieces, not a complete head; [evidence](generation/dev.readme.md#validation) |
| Blender | Shared Blender 4.5.15 LTS, minimal cleanup, preserved UVs/cameras and CUDA renders | Avoid invented anatomy, smoothing and automatic component deletion; holes/lighting/likeness still need review | Actual textured mesh export and three non-flat renders on RTX 4070 SUPER |
| Photo-conditioned head | Pinned GNM 3.0 prior; MediaPipe CPU observations; shared identity with per-photo camera/expression; CUDA autodiff Schur/LM | Coherent prior-driven anatomy, not a calibrated scan; no personalized albedo, rig or likeness approval | Fresh 63-view pipeline 372.859s CPU -> 42.187s CUDA; held-out error slightly better; [evidence](facefitting/dev.readme.md#evidence) |
| Unreal | Installed UE 5.8.4; existing guarded Identity adapter; prefer head-only Custom Mesh as next experiment | Arbitrary GNM topology is not From Template topology. Required Core Data is missing; actual import, conform and shape drift remain unverified; cloud stays opt-in | Real owned-project commandlet loaded Python conforming APIs; doctor reports content separately; [investigation](generation/metahuman-workflow-investigation.md) |
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
Blender/COLMAP and the new CUDA fitter ran locally; UE 5.8.4 metadata and the
MetaHuman Character plugin are installed and Python API exposure was editor-tested
in a NullRHI commandlet. No import/conform or rig validation is claimed. See
[generation evidence and primary references](generation/dev.readme.md) and the
imageprocessing research log. Documentation availability is not proof of local
engine installation or compatibility.

# ailias

Photo-to-MetaHuman tooling: prepare reference images, fit a complete head, inspect
native Unreal assets, and animate an existing rig with supplied narration.
Python coordinates local OpenCV, PyTorch/SciPy, COLMAP, Blender and Unreal workers.

**Current limitation:** native conforming, rigging and animation work technically,
but facial resemblance and skin/hair transfer are below expectations. Native
feature profiles now save/reload visible skin, eyes and native grooms; automatic photo-to-native likeness fitting and
photographic appearance transfer are not complete.

## Visual pipeline

Actual retained images in 160 x 160 thumbnails, with aspect ratio preserved.
These illustrate stage contracts across different jobs, not one fully automated
successful run. **LLM** means a language model; none is required by these stages.

| Input representation | Module / operation | Output representation |
|---|---|---|
| ![Reference portrait](docs/images/pipeline/portrait.png) | **imageprocessing** - Detect/group faces, isolate heads and extract regional shape/colour/iris evidence. **LLM:** no. **GPU:** optional; explicit CPU/CUDA. **Needs:** local OpenCV and prepared detection/parsing/matting weights; MediaPipe for dense inspection; no Blender or Unreal. | ![Head-only cutout](docs/images/pipeline/cutout.png) |
| ![Prepared reference head](docs/images/pipeline/cutout.png) | **facefitting / fit** - Fit shared head identity with per-photo camera/expression. **LLM:** no. **GPU:** configured CUDA; explicit CPU solver available. **Needs:** GNM, MediaPipe and PyTorch/SciPy; Blender for the review scene, not numerical fitting. | ![Neutral fitted geometry](docs/images/pipeline/clay.png) |
| ![Neutral geometry](docs/images/pipeline/clay.png) | **facefitting / appearance** - Project observed colour onto the fitted mesh and approximate scalp hair. **LLM:** no. **GPU:** configured CUDA projection/rendering. **Needs:** reviewed fit and Blender; no Unreal. Colour still contains photographed lighting. | ![Blender appearance preview](docs/images/pipeline/appearance.png) |
| ![Photo-derived preview](docs/images/pipeline/appearance.png) | **Native handoff / features** - Import/conform geometry, apply explicit native skin/eye/groom edits and optionally assemble a rig. **LLM:** no. **GPU:** UE preview/rendering. **Needs:** UE 5.8, MetaHuman content/plugins and compatible assets; no Blender for edits to an existing Character. Photo matching remains incomplete. | ![Latest editable UE appearance draft](docs/images/pipeline/native-appearance.png) |
| ![Existing native character](docs/images/pipeline/native.png) plus narration WAV | **reanimation** - Solve face/head motion from supplied audio, render and encode. **LLM:** no; no speech synthesis. **GPU:** UE rendering; configured NVENC encoding, explicit CPU encoder available. **Needs:** UE 5.8 Animator/MRQ plugins, assembled rig and FFmpeg; no Blender. | ![First and last frames of the resulting reel](docs/images/pipeline/reel.png) |

Image provenance and resize/crop records: [pipeline media](docs/images/pipeline/provenance.json).
The reanimation input thumbnail is historical; the editable appearance draft is
an actual newer UE viewport capture, not a completed rig or matched-camera render.
Its provenance is recorded with the [appearance comparison](docs/facefitting/images/ue-photo-appearance-provenance.json).
These are stage-specific requirements, not simultaneous residency requirements;
tool paths, devices and model preparation are covered by the module guides.

### Resources passed between stages

| Producer | Retained resources | Next consumer / reuse |
|---|---|---|
| `imageprocessing scan / cleanse` | Run manifest, portrait RGB, native-grid head RGB/alpha, source transforms, curation | `facefitting fit` and `generation reconstruct`; aligned analysis images do not replace native pixels |
| `facefitting fit` | Neutral OBJ, identity parameters, observations, cameras, overlays, review scene | `appearance` uses the exact reviewed mesh and fitting evidence |
| `facefitting appearance` | Hair-free OBJ/MTL, photo-colour PNG, coverage, geometry/UV NPZ, scene, approximate hair Alembic | `unreal-import` consumes mesh/material/texture; Alembic still needs validated native alignment/binding |
| `facefitting unreal-import / unreal-conform` | Static-mesh or Character packages, native receipts, camera/landmark evidence | `unreal-features` accepts explicit edits and optional assembly; photo-guided geometry/skin/groom mapping remains incomplete |
| `facefitting unreal-features / unreal-review` | Isolated Character revision, edit/reload receipts, optional runtime assets and neutral review frame | Validate native likeness before animation; failed assembly/inspection does not become an accepted rig |
| Native rig/build | Assembled rigged Blueprint with Face mesh, head bone and facial post-process | `reanimation run` plus an independent mono PCM16 WAV |
| `reanimation` | Performance, sequence/map, verified MP4, first/last PNGs, hashes | Playback/compositing; technical media checks do not certify likeness |
| `generation` - alternative route | Sparse model, estimated cameras, fused samples, textured mesh and cleaned scene | `clean-mesh` / static Unreal import; current real-photo outputs are fragments, not a full head |
| `geometry` - independent validator | Named-landmark/region errors, alignment transform, optional sampled-surface distances | Comparison reports; requires corresponding measurement JSON, not arbitrary OBJ input |

## Module architecture

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 1100}}}%%
flowchart TB
    subgraph MAIN[" "]
        direction TB
        PH["Reference photographs"]:::store --> IMG["imageprocessing: OpenCV DNN<br/>Grouping, head masks and reversible coordinates"]:::svc
        IMG -->|Native RGB, alpha and curated references| FIT["facefitting: MediaPipe + PyTorch / SciPy<br/>Shared head shape, cameras and photo appearance"]:::svc
        IMG -->|Masked hardlinked views| GEN["generation: COLMAP + Blender<br/>Alternative reconstruction and cleanup"]:::svc
        FIT -->|OBJ, texture and camera files| UE["Native handoff: Blender + UE 5.8<br/>Static import and Character conform"]:::ext
        UE -.-> GAP["Missing photo-to-native mapping<br/>Validated likeness, skin and groom matching"]:::unknown
        GAP -.-> RIG["Assembled rigged MetaHuman Blueprint"]:::store
        RIG --> REEL["reanimation: UE Animator / MRQ + FFmpeg<br/>Audio solve, portrait rendering and encoding"]:::svc
        WAV["External narration WAV"]:::store --> REEL
        REEL --> VIDEO["Verified portrait MP4 + diagnostics"]:::store
        JSON["Corresponding landmark / surface JSON"]:::store --> GEO["geometry: NumPy<br/>Alignment and regional errors"]:::svc
        CAP["Core workflow<br/>Capture review, handoffs and baselines"]:::svc
    end
    MAIN ~~~ SUM
    SUM["Figure - Module responsibilities. How to read: top to bottom; blue modules, green resources, purple editor integration, red dashed missing automation. Summary: preparation and animation are separate working paths; native likeness/appearance automation is the gap. Backed by: imageprocessing/cli.py, facefitting/pipeline.py, facefitting/unreal_conform.py, generation/pipeline.py, reanimation/pipeline.py, geometry/metrics.py and workflow.py."]:::caption
    style MAIN fill:none,stroke:none
    classDef svc fill:#cfe3ff,stroke:#2b6cb0,color:#0b2545
    classDef ext fill:#e6dcff,stroke:#6b46c1,color:#2d1b52
    classDef store fill:#d6f5df,stroke:#2f855a,color:#12351f
    classDef unknown fill:#fde2e2,stroke:#c53030,color:#611a15,stroke-dasharray:4 3
    classDef caption fill:#f7f7f7,stroke:#999,color:#222,font-size:16px
```

**Backed by:** [module ownership and file-based communication](docs/dev.readme.md#architecture).
Core capture/handoffs are a separate bookkeeping workflow, not an automatic
consumer of every experimental job. Geometry validation requires supplied correspondences.
For alternative face-first routes, repositories and small-model tradeoffs, see
[photo-to-MetaHuman research](docs/imageprocessing/photo-to-metahuman-research.md).

## Quick setup

Use Python 3.12+ in the shared workspace environment; preserve its existing
CUDA/OpenCV/Torch installation.

```powershell
Set-Location C:\projectWorkspace\ailias
.\scripts\setup-gpu.ps1
& C:\projectWorkspace\.venv\Scripts\Activate.ps1
python -m ailias --help
```

Model/tool preparation is explicit. Scans, status checks and imports do not
silently install models or replace the shared runtime.

| Task | Command / guide |
|---|---|
| Prepare analysis models | `python -m ailias imageprocessing models subject` |
| Analyze photos | `python -m ailias imageprocessing scan subject C:\photos --device cuda` |
| Sample video photos + timed expression/audio evidence | `python -m ailias facefitting video-evidence subject C:\recordings` - default 1 FPS/1080p; [video identity workflow](docs/facefitting/video-identity-investigation.md) |
| Create fresh Blender and native heads from sampled photos | `python -m ailias facefitting video-create subject --evidence C:\data\sampled\manifest.json --editor $ue --project $project` - fresh textured Blender scene/export, native appearance and portrait; no previous personal head/fit, no likeness approval |
| Review and select references | [Imageprocessing commands](docs/imageprocessing/README.md#usage) |
| Fit / texture / import a head | [Facefitting commands](docs/facefitting/README.md#usage) |
| Animate a compatible existing rig | [Reanimation commands](docs/reanimation/README.md#usage) |
| Reuse one Unreal editor | `python -m ailias generation unreal-session start --editor $ue --project $project` - [lifecycle, configuration and checks](docs/generation/README.md#reusable-unreal-editor) |
| Inspect retained resources | `python -m ailias cleanup --alias subject` - dry run; `--apply` deletes eligible remnants |
| Record capture and external handoffs | [Core workflow](docs/workflow.md) |

## Configuration

Global `--config` selects `config\settings.json`; `--data-root` overrides its
data directory. Module flags select their own JSON policies. Per-run overrides,
where implemented, take precedence without rewriting those files.
See each module guide below for defaults, units and parameter meanings.

## Current UE appearance - likeness not approved

| Selected reference | Earlier editable UE appearance (f6b61b60) |
|---|---|
| ![Selected glasses-free reference](docs/images/pipeline/appearance-reference.png) | ![Native skin, eyes, brows, fringe, moustache and stubble](docs/images/pipeline/native-appearance.png) |

Revision `f6b61b60` refines native skin roughness, complexion, undertone and lip
colour. It saves and independently reloads with the owner-approved beard and
moustache preserved, alongside unchanged eyes, brows and hair. The skin pass uses
a lower-mark native texture, not a blurred photograph. This revision's geometry is unchanged;
hair is still straighter than the reference and skin reflectance is not
photometrically calibrated. This is not full-likeness approval.
[Actual photo/UE comparison](docs/facefitting/images/ue-photo-appearance-comparison.png).
[Skin before/after](docs/facefitting/images/ue-skin-before-after.png).

The latest fitted-head runtime `35ed200a` fixes the baked checkerboard under
verified SM6. Its complete 24.77-second speaking draft was decoded and visually
inspected, reusing one working editor for assembly, portrait and video. Geometry
and supplied narration are unchanged. Complexion still appears lighter than the
reference, and likeness remains unapproved.
[Actual run, timings and remaining work](docs/reanimation/pipeline-optimization-investigation.md#native-fallback-and-real-pipeline-follow-up).

## Retired Blender baseline

On November 18, the owner requested deletion of the selected Blender baseline.
Its `hairfree-matched-camera.blend` scene and active `working-baseline.json`
selector have been removed. Original sampled photos and native Unreal Characters
remain intact. New full runs must create fresh downstream outputs rather than
restore or initialize from the deleted scene.
See [baseline resources](docs/facefitting/README.md#current-working-baseline).

## Documentation

| Document | Dev guide | Purpose |
|---|---|---|
| [Core workflow](docs/workflow.md) | [Development](docs/dev.readme.md) | Capture, durable state, handoffs, global configuration and cleanup |
| [Imageprocessing](docs/imageprocessing/README.md) | [Internals](docs/imageprocessing/dev.readme.md) | Grouping, head extraction, curation and colour policy |
| [Facefitting](docs/facefitting/README.md) | [Internals](docs/facefitting/dev.readme.md) | Head geometry, appearance and native Character handoff |
| [Generation](docs/generation/README.md) | [Internals](docs/generation/dev.readme.md) | Alternative COLMAP reconstruction and Blender cleanup |
| [Geometry](docs/geometry/README.md) | [Internals](docs/geometry/dev.readme.md) | Corresponding-landmark and sampled-surface comparison |
| [Reanimation](docs/reanimation/README.md) | [Internals](docs/reanimation/dev.readme.md) | Existing rig plus WAV to portrait video |
| [Pipeline latency investigation](docs/reanimation/pipeline-optimization-investigation.md) | [Reanimation internals](docs/reanimation/dev.readme.md) | Persistent Unreal proposal, standalone-library limits and measured optimization priorities |
| [Decisions](docs/analysis.md) | [Open work](docs/todo/README.md) | Tradeoffs, limitations and acceptance criteria |

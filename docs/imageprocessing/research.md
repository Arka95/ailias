[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Image preparation and likeness metrics research

Primary sources checked November 16, 2026. Images remain local; research queries
did not include user photos or embeddings.

## Photo-to-MetaHuman investigation

[Face-first workflows, repositories and small models](photo-to-metahuman-research.md)
(November 17, 2026) compares native fitting, neural reconstruction, official
handoffs and commercial alternatives. It assigns reusable photo feature
extraction to imageprocessing and native control/UV translation downstream.
Small VLMs are optional descriptor helpers, not geometry or DNA generators;
concurrent UE/model memory has not been measured.

## Model choices

| Task | Choice | Why / tradeoff | Primary source |
|---|---|---|---|
| Face boxes and five landmarks | YuNet, OpenCV native CPU | Small model, integrates directly with recognition; profile/occluded faces can be missed | [OpenCV face tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html) |
| Candidate identity grouping | SFace normalized embedding cosine | Separate identity signal from physical geometry; thresholds vary by dataset and are not probabilities | Same OpenCV tutorial |
| Visible-person extraction | PP-HumanSeg | Small CPU model; semantic person foreground, not per-person instance segmentation or hair matting | [Official model and preprocessing](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/human_segmentation_pphumanseg) |
| HEIC decoding | pillow-heif | Integrates with Pillow; decode full primary image with thumbnails disabled | [Official plugin documentation](https://pillow-heif.readthedocs.io/en/latest/pillow-plugin.html) |
| Orchestration | Python with native C++ OpenCV and NumPy kernels | Fits existing CLI and image ecosystem; no claim Python is compulsory or that a native rewrite is justified | Verified local adapters and installed runtime |

Model provider: `opencv/opencv_zoo`, immutable revision
`48b10fa505aeb51fbd127fc60a5c660040e823b7`. Models total 45,092,880 bytes
before licenses. SHA256/size were taken from this revision's Git LFS pointers
and are verified before inference.

YuNet is distributed with MIT terms; SFace and PP-HumanSeg with Apache-2.0
license files in their model directories. Downloaded licenses and their hashes
are retained beside the weights. This module does not redistribute weights in Git.

OpenCV's tutorial reports a 0.363 cosine threshold on LFW, with different
thresholds on other evaluation sets. That is not a universal alias threshold.
The implemented 0.50 complete-link gate is an intentionally conservative product
default; it needs calibration on reviewed subject photos.

Generative face restoration and aggressive denoising were rejected for
reconstruction preparation: they can replace missing evidence with generated or
smoothed detail. The implementation therefore limits cleansing to format/color/
orientation handling, duplicate filtering, quality measurement, crops and masks.
This is an engineering choice, not a claim that one model can recover unseen
anatomy or make poor inputs reconstruction-ready.

## What facial similarity can mean

| Signal | Implemented use | Interpretation / limitation |
|---|---|---|
| Identity cosine | Face alias candidates | Higher means embedding agreement; not facial dimensions or 3D likeness |
| 2D landmark normalized RMSE | Five projected points after similarity alignment | Lower means projected shape agreement; use matched pose, expression and camera |
| 3D landmark error | Rigid/no-reflection alignment, errors in mm and interocular-normalized RMSE | Corresponding landmarks quantify sparse physical geometry; dense jaw/ear shape may remain unobserved |
| Sampled Chamfer / Hausdorff / p95 | Bidirectional nearest-sample distances after landmark alignment | Measures supplied 3D samples only; density, region crop and sampling affect values |
| True scan-to-mesh error | Research reference, not implemented | NoW uses landmark-initialized alignment and closest mesh-surface distance, not just vertices |
| Region-wise surface evaluation | Future | REALY shows why region definitions and correspondence/alignment affect conclusions |
| ICP fitness/inlier RMSE | Candidate future Open3D adapter | Registration diagnostics depend on correspondence threshold and overlap; not a universal likeness score |

Primary references:

- [NoW benchmark and evaluation procedure](https://now.is.tue.mpg.de/)
- [REALY paper](https://arxiv.org/abs/2203.09729)
- [Open3D ICP registration](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
- [SciPy Procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html)
- [SciPy directed Hausdorff](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html)

The implementation uses a small NumPy SVD alignment instead of adding SciPy or
Open3D just for sparse landmarks. Unlike unconstrained Procrustes that can allow
reflection, mirroring is forbidden. `rigid` is the 3D default and preserves
physical size; explicit `similarity` fitting removes size, which can hide a
scale error. No arbitrary 0-100 similarity percentage or pass threshold is invented.

For actual MetaHuman validation, compare neutral-expression geometry in the
same metric units and facial region mask. Use calibrated reconstructed geometry
as reference, not a photo embedding. Matched-camera silhouette/reprojection
checks and animation testing remain independent evidence.

## Local validation record

Initial CPU environment on Python 3.12.10: Pillow 12.3.0, NumPy 2.4.6,
OpenCV headless 4.14.0.94 and pillow-heif 1.8.1. Existing NumPy/Pillow were reused;
no Torch/CUDA stack was installed for that initial run.

A first three-image local integration smoke run produced portrait PNGs, body RGBA
PNGs and masks with no file errors. A subsequent CPU run processed all 192 files:
210 face detections, 9 no-face images, no file errors, 176 visible-body exports,
20 unmatched foreground associations and 14 ambiguous touching-subject cases.
121 face crops passed the configured quality rules before identity ambiguity.
These are processing counts, not manually measured accuracy.

Run: `20261116T100526Z-47ce6ae5`, under
`data\arka\imageprocessing\runs`. Its **173 groups / 141 ambiguous detections**
exposed an initial greedy clustering bug: uncertain singletons were becoming
competing anchors and causing further fragmentation. **Do not treat that run's
alias grouping as validated.** The current code instead uses agglomerative
complete-link clustering and post-clustering ambiguity flags; synthetic
regressions cover the cascade, bridge and co-occurrence cases. Selection refuses
the superseded grouping revision. Superseded smoke/CPU outputs were subsequently
removed during explicitly requested data cleanup; the useful full GPU run remains.

The owner requested a reboot pause, then confirmed readiness on November 16,
2026. The rebooted RTX 4070 SUPER was detected with driver 610.88 and 12,282 MiB.
System CUDA Toolkit 13.5 was already installed; Thea's existing environment also
had OpenCV 4.13.0.90 with CUDA DNN enabled. A redundant ONNX Runtime installation
was stopped rather than replacing that working stack.

The current GPU implementation uses the one shared workspace environment,
`C:\projectWorkspace\.venv`, a zero-copy junction to the existing Thea environment.
Only missing Pillow/HEIC and the editable ailias package were added there; the
extra app-local environment was removed. CUDA/OpenCV/NumPy were reused, not
reinstalled or copied. A three-photo actual smoke run
`20261116T103725Z-39eacc17` verified completed CUDA inference graphs for all models:
118 YuNet, 92 SFace and 191 PP-HumanSeg graph-layer entries marked `CUDA/CUDA`.
No CPU layers were reported in these model graphs. These counts verify backend
assignment after inference, not speedup or a claim that image I/O/cropping/grouping
run on GPU. The existing CUDA stack and separate CPU ONNX Runtime were unchanged.

Primary OpenCV API/implementation references:

- [DNN network backend/target dump](https://docs.opencv.org/4.x/db/d30/classcv_1_1dnn_1_1Net.html)
- [YuNet postprocessing](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_detect.cpp)
- [SFace alignment and RGB input](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_recognize.cpp)

Full GPU run `20261116T103801Z-d7f8b6f6` processed all 192 photos: 210 detections,
9 no-face images, no file errors, 55 conservative candidate groups and 24 eligible
unambiguous portraits. There were 145 identity-ambiguous detections; 121 crops
passed quality rules before ambiguity exclusions. Body outputs: 176 visible,
20 unmatched and 14 touching-subject ambiguity cases.

The overview, a group's contact sheet and a body extraction were visually reviewed.
Groups still over-split viewpoint changes; some detections are artwork/statues.
These counts verify processing, not identity accuracy or reconstruction readiness.
No real 3D reconstruction, MetaHuman conversion or likeness certification was run.
The retained manifest was migrated to original-file references and shared generated
assets; original photos were hash-verified before redundant copies were removed.

After switching to the shared interpreter, a fresh three-photo CUDA smoke run
completed with two faces, two visible-body outputs and no file errors. Generated
asset count stayed at 562: existing crops/masks were reused without replaying
inference. The temporary validation run was then pruned. All 192 retained source
references, 562 asset hashes and group contact-sheet paths verified successfully.
All 72 functional units passed in the shared environment, followed by a focused
locked-directory regression after the final cleanup-report adjustment.

Requested cleanup reduced generated data from 3,324,003,748 to 1,167,994,803 bytes
(about 2.16 GB removed), retaining the full GPU run and model provenance.
An empty superseded run directory, `20261116T100507Z-275b19a8`, remains held open
by Windows; its files are gone, and cleanup reports it in `retained_empty_runs`.
The app-local venv and empty starter alias were removed.

## Curated portrait filtering

Research checked November 16, 2026 against primary sources:

| Candidate | Decision / tradeoff | Evidence |
|---|---|---|
| Existing normalized SFace features + cosine | Chosen for exemplar comparisons; reuses the already pinned recognizer and CUDA stack without more models | [OpenCV 4.13 face-recognition tutorial](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html), [SFace paper](https://arxiv.org/abs/2205.12010) |
| Single averaged face/centroid | Not chosen; retaining individual reviewed examples preserves pose and subject variation instead of collapsing the gallery | Engineering choice; no claimed centroid-versus-gallery accuracy benchmark |
| Five-landmark geometry error | Not used as a validity/identity classifier; projected sparse shape does not establish a correct human portrait | Existing geometry module's explicitly limited contract |
| MagFace / QMagFace quality-aware features | Considered, deferred: separate trained representations and model-specific quality behavior, not a property to assume for normalized SFace | [MagFace](https://arxiv.org/abs/2103.06627), [QMagFace](https://arxiv.org/abs/2111.13475) |
| Score thresholds with an abstention band | Chosen rather than forced binary deletion; unknown and conflicting examples require review | [Decision scores versus decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html); implementation does not add scikit-learn |

OpenCV reports different recognition thresholds for different datasets; its
LFW cosine value of 0.363 is not a universal portrait-quality or alias threshold.
SFace is an identity representation, not a general invalid-content/blur detector.
MagFace learns quality-related embedding magnitudes with its own training loss;
the magnitude of an already normalized SFace vector is not a quality score.

The implemented classifier is therefore deliberately limited: exact reviewed
keep/delete decisions are remembered, while **new** crops need agreement between
nearest kept/rejected examples and existing quality gates. Strong rejected
matches with weak positive support are suppressed; unknown or conflicting crops
are separated under `assets\review`. Human-kept crops are preserved even when
they fail downstream reconstruction-quality gates. Automatic predictions never
become new labels without explicit human review.

The owner had retained **182** and removed **28** of the original 210 portraits
when the reviewed profile was captured. Those 210 feature/quality records were
reused from the completed manifest; no deleted portrait was regenerated to
learn its rejected label. Labels describe valid/invalid crops, not verified
identity names.

With each entire source photo held out and exact-crop overrides disabled,
the diagnostic classified kept examples as 112 accept / 70 review / 0 reject,
and removed examples as 0 accept / 26 review / 2 reject. This illustrates both
the safety preference and limited generalization: similarity alone does **not**
automatically identify every invalid portrait. These are diagnostics on the
reviewed collection, not an independent future-data accuracy claim.

Actual CUDA replay `20261116T111557Z-809037c5` processed all 192 original photos:
210 detections, **182 accepted and 28 rejected**, no file errors. All kept hashes
remained unchanged; none of the 28 removed portrait paths was recreated.
Rejected detections exported no portrait/body/mask products. There were 30
candidate groups and 24 quality/ambiguity-eligible faces; curation does not certify
identity or reconstruction readiness.

Final-code CUDA checks on kept/rejected source photos also passed, including
verified execution of all three models; their temporary run folders were removed.
All 192 source hashes, accepted output references and the profile fingerprint
verified successfully. **86 functional unit tests passed**, and `git diff --check`
reported no whitespace errors. Units use synthetic images/fake inference; the
real GPU replay is separate opt-in validation, not a held-out accuracy evaluation.

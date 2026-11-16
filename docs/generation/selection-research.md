[Back to REPO_ROOT](../../README.md) | [Generation](README.md)

# Multi-view selection and incomplete facial evidence

Primary sources checked November 16, 2026. Research informs the choices below;
it does not imply these papers' pretrained models or reconstruction accuracy
have been incorporated into ailias.

## Evidence and decisions

| Source | Relevant finding | Choice in ailias | Limit |
|---|---|---|---|
| Deng et al., [Accurate 3D Face Reconstruction: From Single Image to Image Set](https://arxiv.org/abs/1903.08527), CVPR Workshops 2019 | Multi-image shape aggregation can exploit complementary information rather than treating every photo as interchangeable | Separate seed-subject association from view categories; keep complementary directions and explicit observation provenance | Implemented selection is a heuristic, not their learned aggregation network |
| Sanyal et al., [RingNet](https://arxiv.org/abs/1905.06817), CVPR 2019 | Identity shape should be distinguished from image-dependent pose/expression | Frozen seed anchors, cross-group identity evidence, independent pose bins; never use a pose cluster as a confirmed identity | No RingNet/FLAME weights, expression neutralization or template fitting is implemented |
| [OpenCV PnP](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html) | Pose estimation needs corresponding 3D/2D points and camera intrinsics | SQPnP with five stored landmarks, three focal hypotheses, positive-depth checks and normalized reprojection/focal-sensitivity confidence | Unknown intrinsics and a generic face make angles approximate; five points cannot describe detailed facial shape |
| [MediaPipe canonical face](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/modules/face_geometry/data/canonical_face_model.obj) and [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | Canonical geometry supplies semantic correspondences; the full task supports dense landmarks, expressions and transformations | Five numerical proxy points derived from the tagged canonical asset; retain source-space landmarks alongside recovered cameras | MediaPipe runtime/dense model is **not installed or executed**; these proxies are not a full-head template |
| Tran et al., [Extreme 3D Face Reconstruction: Seeing Through Occlusions](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_Extreme_3D_Face_CVPR_2018_paper.pdf), CVPR 2018 | Visibility and symmetry can provide approximate support for self-occluded regions | Proposed confidence-tagged symmetry fallback only where actual observations are absent | Not implemented; real asymmetry must not be overwritten and mirrored detail is not measured evidence |
| Selvaraju et al., [OFER](https://openaccess.thecvf.com/content/CVPR2025/html/Selvaraju_OFER_Occluded_Face_Expression_Reconstruction_CVPR_2025_paper.html), CVPR 2025 | Occluded geometry is ambiguous; multiple plausible completions can exist | Proposed shape-prior completion must label inferred areas and preserve uncertainty | No OFER/diffusion model adopted; a plausible completion cannot certify the owner's hidden anatomy |
| Feng et al., [DECA](https://deca.is.tue.mpg.de/), SIGGRAPH 2021 | Shape, person-specific detail, expression, albedo and illumination are distinct estimation targets | Proposed shared identity shape with per-photo expression/light rather than baking expression or shadows into anatomy/skin | Architectural reference only; no DECA model installed |
| Zielonka et al., [MICA](https://arxiv.org/abs/2204.06607), ECCV 2022 | Perspective creates depth/scale ambiguity; metric face prediction needs additional learned or measured information | Report relative proportions until owner-specific scale is supported; validate inferred metric dimensions independently | A model's metric coordinate system is not proof of accurate owner measurements; no MICA model installed |
| Azinovic et al., [High-Res Facial Appearance Capture from Polarized Smartphone Images](https://arxiv.org/abs/2212.01160), CVPR 2023 | Controlled polarized capture and inverse rendering can separate diffuse/specular reflectance and surface detail | Prefer lighting-separated albedo to a single average photographed skin colour; controlled capture is a later quality option | Not implemented; casual existing photos are not equivalent to their polarized capture protocol |
| Feng et al., [Towards Racially Unbiased Skin Tone Estimation via Scene Disambiguation](https://arxiv.org/abs/2205.03962), ECCV 2022 | Full-scene illumination information helps disentangle skin reflectance from illumination | Preserve original scene evidence even when deriving background-free portraits; do not normalize all skin to one colour | Research reference only; no TRUST model adopted |
| [OpenCV GrayworldWB](https://docs.opencv.org/4.13.0/d7/d71/classcv_1_1xphoto_1_1GrayworldWB.html) and Ke et al., [MODNet](https://arxiv.org/abs/2011.11961), AAAI 2022 | White balance has scene assumptions; portrait matting estimates foreground opacity rather than only binary person labels | Proposed conservative colour correction and dedicated portrait alpha mattes as separate versioned cleansing products | Neither is implemented for portraits; existing sRGB conversion/body masks/face ellipses must not be presented as these features |

The explicit frontal preference is a product choice, not a universal optimum
claimed by these papers. Frontal observations anchor facial layout; side views
remain valuable complementary constraints. Exact quotas and association thresholds
are inspectable heuristics, not calibrated identity probabilities.

## Implemented association and coverage

The original complete-link imageprocessing groups remain immutable candidate
clusters. Generation builds a separate subject hypothesis starting at the
requested seed group:

1. Use only currently human-kept portraits with unchanged original/crop hashes.
2. Associate an outside face only with support from at least two distinct seed
   source photos at the existing cosine threshold, adequate seed-centroid
   similarity, and a margin over available co-occurring competing faces.
3. Keep the seed fixed: associated candidates cannot recursively recruit weak
   bridge matches. Faces co-occurring with the seed and competing proposed faces
   from one source are excluded/abstained rather than merged.
4. Categorize approximate front, left/right oblique, left/right profile,
   above/below-camera and unknown views independently of identity grouping.
5. Cover available directions, then allocate by weights: front 0.50, each
   oblique 0.12, each profile 0.07, above/below 0.04 each, unknown 0.04. Missing
   buckets redistribute capacity; frontal-first is a preference, not fabricated
   coverage. Within bins, prefer no quality warnings, reliable pose, face size
   and detail. Maximum input count is now 64.
6. Preserve a bounded set of still-eligible, previously registered photo anchors
   from one hash-verified camera artifact with matching run/group/curation.
   This protects some known geometric tracking evidence; it does not guarantee
   that the widened view graph is connected. The correction is implemented in
   the working tree but has not had a native reconstruction replay.

Strict generation still rejects quality-warning portraits. The explicit PoC
profile can use human-kept exposure/blur/small-face warnings at reduced observation
weight, but excludes faces smaller than 64 native pixels and crops flagged as
containing another face. Unknown poses remain eligible, not automatically deleted.
This does not replace neutral-expression, eyewear or occlusion assessment.

Source-space five-point observations, view uncertainty, quality warnings,
candidate group and heuristic selection weight are saved per input. Recovered
camera records retain that metadata. **COLMAP does not consume these as semantic
face-fitting constraints.** `semantic_facial_constraints_used` remains false.
Selection weights are not silently presented as native SfM objective weights.

## Mesh detail and missing regions

The PoC now uses a dense-image side of 1,200 and Poisson depth 10 rather than
800/8. Raw fused `observed-surface.ply` is retained with the camera/sparse evidence,
including on subsequent surface-quality failure. Texture/camera provenance stays
intact and transient stereo maps/databases still finalize.
Retention currently happens after successful native meshing, before the
mesh-size check; a failure inside the meshing command can still discard the
uncommitted fused cloud. Do not interpret retention as an all-failure guarantee.

These settings allow more detail **where observations support it**; they do not
add missing anatomy. A high face count from Poisson interpolation is not proof
of high facial information. A coherent detailed head still needs a template
fitter with dense semantic correspondences, per-view visibility/expression
handling and region-specific evidence/uncertainty. That fitter is not implemented.

Recommended completion order for that future fitter:

| Region evidence | Treatment | Provenance |
|---|---|---|
| Reliable matching observations in several views | Fit shared identity shape to the observations; keep per-image pose/expression separate | Observed/estimated with contributing view IDs |
| Evidence on only one side | Weak, reversible symmetry prior where no contradictory observation exists | Inferred from opposite side, reduced confidence |
| Sparse evidence near known surface | Regularized template deformation, preserving plausible topology and local shape | Prior-supported, not newly measured |
| No useful evidence, especially the rear head | Retain a generic template/shape prior; do not extend arbitrary photogrammetry fragments into a claimed identity | Unobserved/generic; request recapture only if needed |

[MakeHuman's core basemesh assets are CC0](https://www.makehumancommunity.org/about/license.html)
and are a candidate for a complete baseline without adopting its AGPL application
code. FLAME-based research is another architectural reference, not an installed
or licensed asset dependency. A specific baseline/fitter must be validated before
claiming missing-region completion.

### Optional owner context

Age, self-described gender/background, height and weight are optional context,
not replacements for photo evidence. A future backend may accept applicable
template initialization controls, but measurements/observations must dominate.
Height/weight do not directly determine facial landmarks; country/background
must not be converted into stereotyped facial dimensions. No demographic face
morphing or additional profile storage was added to this selection change.
Until a backend can meaningfully consume such controls, no new parameters are
required and nothing is inferred from appearance.

## Actual selection audit

For run `20261116T111557Z-809037c5`, seed `face-001`, permissive profile:

| Measure | Result |
|---|---|
| Human-kept portraits | 182, unchanged |
| Candidates after hard exclusions | 179 |
| Seed/associated hypothesis | 42 seed + 128 supported cross-group candidates |
| Uncertain associations | 9, retained in a separate review contact sheet |
| Chosen inputs | 64 from 19 original groups, versus 31 from one group previously |
| Approximate chosen bins | 38 front, 10 left-oblique, 10 right-oblique, 1 left-profile, 1 right-profile, 1 above-camera, 0 below-camera, 3 unknown |

Bins are estimates, not human-verified poses or proof of identity. The contact
sheets expose them for correction. Retrying unchanged selection reuses the same
hardlinked input view/masks; source photos are not copied.

The initial widened reconstruction, job `20261116T130010Z-e60844a8`, verified
only two image pairs with a largest connected component of two, and blocked
before mapping. It removed 17,350,656 transient bytes with zero cleanup errors.
No denser head or multi-view template fit was produced. Later anchor-preserving
selection retains the same bin totals above, but native replay remains pending.

For compact explanations by owner module, see [image observations](../imageprocessing/dev.readme.md#research-decisions-in-plain-language),
[generation](dev.readme.md#research-decisions-in-plain-language) and
[geometry measurements](../geometry/dev.readme.md#research-decisions-in-plain-language).

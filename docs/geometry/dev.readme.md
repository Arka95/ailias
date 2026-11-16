[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Geometry internals

`metrics.py` contains pure NumPy transformations and metrics.
`compare_geometry(reference, candidate, fit="rigid")` accepts the guide's JSON-like
objects and returns JSON-safe numeric errors. NumPy is imported lazily.

Alignment solves the least-squares SVD rotation with determinant +1, explicit
translation and optional positive uniform scale. No reflection or nonrigid warp
is allowed. Regional errors reuse the global alignment rather than separately
fitting each region and hiding anatomical offsets.

`nearest_distances` computes exact Euclidean nearest-sample distances in
128-point chunks, avoiding the full pairwise tensor. The 10k input cap bounds
memory/work for this initial implementation; runtime remains quadratic.
For larger meshes, prefer a measured spatial-index / point-to-triangle adapter,
not a silent subsampling or metric change.

`comparison.py` persists input snapshots/reports and separates photo-embedding
cosine from five-landmark 2D errors. Face comparison uses named correspondences
from YuNet's fixed five-point ordering. It never uses the identity cosine as a
geometry pass/fail criterion. `cli.py` owns the explicit comparison command.

Validation uses synthetic transformed shapes, changed nose geometry, metric
unit conversion, mirrored faces, scale-preserving versus scale-invariant fits,
known sampled-surface distances and invalid/degenerate inputs. No actual 3D
face reconstruction or MetaHuman comparison has been executed.

```powershell
python -m unittest discover -s src\ailias -t src -v
```

## Research decisions in plain language

Updated November 16, 2026. **This module compares supplied geometry; it does not
yet extract anatomical dimensions from the photos or fit a head.**

### 1. Problem Statement

Recognition similarity, landmark agreement and real-world dimensions answer
different questions. A face can match five projected points while having the
wrong nose depth. A model labelled "centimetres" is not necessarily calibrated
to the owner; declaring units cannot recover missing scale.

### 2. Implementation

| Decision | Why / limit |
|---|---|
| Keep recognition and geometry scores separate | Implemented: SFace cosine is not a percentage of anatomical likeness; five-point error cannot certify ears, jaw or cheekbones |
| Preserve scale by default; forbid mirroring | Implemented: rigid alignment moves/rotates a model without resizing. Optional similarity alignment also resizes, so can hide a size error; neither permits reflection |
| Use supplied regions under one global alignment | Implemented: moving each feature independently could conceal incorrect placement. Surface errors compare sampled points, not the actual mesh triangles |
| Define measurements on the fitted 3D head | Proposed: record exact endpoints/axes for face height/width, eye spacing/opening, nose width/projection and ear dimensions. "Width/height/depth" alone is ambiguous, particularly for curved surfaces |
| Separate relative proportions from millimetres | Proposed: use trusted scale evidence, such as a measured facial reference or calibrated capture, before claiming physical dimensions. MICA is a learned metric-shape reference, not measurement ground truth or an installed dependency |
| Attach evidence to every reported dimension | Proposed: contributing views, definition, units, observed/inferred/generic status and uncertainty. Hidden eyeball/skull/neck regions must not receive invented measurements |

### 3. Success Criteria

A future dimension report must distinguish measured evidence from model inference
and reproduce known synthetic dimensions after camera/scale transformations.
Validate against independent measurements or scans before claiming accuracy.
Current tests validate the comparison mathematics, **not anatomical accuracy**.
No calibrated millimetre tolerance or single likeness percentage has been chosen.

### 4. Task Breakdown

After generation supplies a fitted template: define named landmarks and reference
planes; add dimension extraction with explicit scale/provenance; validate visible
regions and flag unobserved ones. Keep skin/albedo estimation in generation, not
in a geometry similarity score. Body height/weight are not face-scale calibration.

### 5. Reference Links

Code: `metrics.py:align` / `compare_geometry`,
`comparison.py:compare_detected_faces`.
[Existing metric evidence](../imageprocessing/research.md#what-facial-similarity-can-mean),
[MICA and depth/scale ambiguity](https://arxiv.org/abs/2204.06607),
[MediaPipe relative-depth model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf),
and [proposed fitting/completion decisions](../generation/selection-research.md).

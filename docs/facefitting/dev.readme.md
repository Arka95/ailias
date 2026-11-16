[Back to REPO_ROOT](../../README.md) | [Usage](README.md)

# Head fitting internals

| Module | Responsibility |
|---|---|
| `settings.py`, `config\facefitting.json` | Explicit device, mode counts, fitting budgets and regularization |
| `model.py`, `config\facefitting-models.json` | Pinned public downloads, retained license provenance, safe numeric NPZ loading, surface correspondence checks |
| `observations.py` | CPU MediaPipe 478-point predictions, expression scores, foreground checks and reversible source coordinates |
| `fitting.py` | Camera initialization, normal-based visibility heuristic, region balance, held-out split, explicit CPU reference and shared evaluation |
| `cuda.py` | Batched CUDA deformation/projection/loss, forward-mode derivatives and block-sparse Levenberg-Marquardt with a Schur complement |
| `pipeline.py` | Fresh jobs, immutable dependencies, candidate selection, all-view refit, OBJ/overlays and render invocation |
| `blender_worker.py` | Prior/fitted neutral-clay views and independent review scene, not fabricated SfM output |

## Research decisions in plain language

| Question | Decision and reason | Limitation / alternative |
|---|---|---|
| Why not continue COLMAP? | The cleansed 64-photo set produced only three verified match pairs; a known head shape supplies coherent anatomy when photo triangulation fails | A prior can look plausible without being accurate. Native reconstruction remains separately available |
| What is GNM? | A learned statistical model: template plus weighted identity/expression shape changes. Use pinned Google GNM 3.0 head assets, whose official repository licenses supplied weights and code Apache-2.0 | It is not a text-to-image model. Unseen anatomy is inferred through model correlations, not measured |
| Why not the previously investigated 3DDFA/Basel assets? | Do not assume academic-use-only model terms cover this use; choose the permissively licensed GNM prior instead | Revisit another model only with verified terms and measured benefit |
| Where do image features come from? | MediaPipe predicts 478 points and expression scores. Use the first 468 for the available surface correspondence; retain the ten iris points as observations, not invented GNM anchors | Predicted relative Z is not metric depth. No per-point visibility/confidence is supplied by this adapter |
| Is the community correspondence trustworthy? | Verify numeric weights, indices, triangle membership and active skin-exterior contributions locally. Reject invalid rows by geometry, not landmark IDs | Two active rows fail exterior support and are excluded. Other mapping errors remain possible; author accuracy claims are not independent validation |
| Why separate identity, pose and expression? | One identity must explain many photos while cameras and expressions vary independently. Otherwise a smile or camera tilt can become permanent anatomy | The decomposition is approximate; expression bases can absorb identity differences |
| Why fit in source coordinates? | Analysis alignment is reversible; transform points back to the EXIF-oriented source before perspective fitting | A centered crop's center is not necessarily the camera principal point. Focal length and absolute scale remain uncalibrated |
| What counts as visible? | Require foreground alpha, exterior support and a front-facing prior normal; balance constraints across facial regions | This is not a full z-buffer, glasses detector or proof of visibility. Occluded points can still be predicted incorrectly |
| How is the winner chosen? | Hold out every fifth face-ID-sorted observation within each initial dense view category. Compare category-balanced mean errors, require at least 1% improvement and at most 5% frontal regression | The held-out cameras/expressions are still fitted. This is a conditional 2D check, not independent 3D scan accuracy |
| Why not the overall median? | Many frontal views can hide meaningful oblique improvements. Keep the median as a diagnostic, but balance view categories for selection | The initial trial failed the median-only criterion (0.95% gain); balanced evaluation showed 9.42% gain with 0.72% frontal regression. Historical failed report is preserved |
| What makes it slow? | Initial SciPy code repeatedly evaluated the full batch to estimate derivatives numerically. About 313 of 326 seconds were numerical fits/validation | Merely queuing more CPU work does not accelerate this calculation. First remove repeated derivative work; do not weaken accuracy gates |
| Where should CUDA be used? | Keep fit data, deformation, projection, robust loss and automatic differentiation on one explicitly selected GPU; process every eligible view as one batch | This is a different optimizer over the same objective. Compare quality as well as runtime. I/O, tiny camera initialization and Windows MediaPipe remain CPU work |

Zero joint rotations use the GNM bind-pose linear shape terms; this module does
not implement articulation/skinning or claim an expression rig. The neutral
export deliberately omits per-photo expression coefficients.
FP32 CUDA computation is explicit; scalar diagnostics and final products are
transferred back to CPU. GPU failure is reported rather than silently selecting
CPU or another device. No GPU percentage or speedup is inferred from tensor size.

The CUDA optimizer uses smooth bounded parameters and automatic differentiation,
whereas the CPU reference uses sparse finite-difference least squares. Both use
the same Huber-like landmark objective, identity/expression priors, positive-depth
penalty and publication checks. There is no background queue service, parallel
candidate execution or cross-run inference replay cache yet.

`jacfwd` and `vmap` compute the per-view Jacobian blocks together. A Schur
complement eliminates independent camera/expression blocks before solving the
shared identity update; batched CUDA Cholesky solves the local blocks. This avoids
finite-differencing the full batch for every parameter. Adaptive damping rejects
steps that worsen the same objective.

The shared environment already contained TorchVision 0.28.0+cu126, which requires
Torch 2.13.0. The installed/pinned Torch 2.13.0+cu126 preserves that pairing;
system CUDA 13/OpenCV was not replaced. Same-process OpenCV CUDA GEMM and
TorchVision CUDA NMS both passed. The initially attempted 2.9.1+cu130 install was
superseded, not retained as a second environment.

## Evidence

Authorized CPU baseline `20261116T145947Z-57cdcc65`:
64 curated inputs, 63 usable dense faces, 53 training / 10 held-out identity
views. Candidate 96 wins over candidate 48. Category-balanced held-out error
falls from 0.02769366 to 0.02508408 of outer-eye-corner span (9.423% reduction).
Frontal mean rises from 0.02120392 to 0.02135557 (0.715% regression).
The final all-view refit converges in ten evaluations; its median error is
0.02245693 of eye span, **not a held-out score**.

The generated head has finite vertices and no collapsed triangles; area-ratio
1st/99th percentiles are 0.8315/1.2630. Self-intersections are not checked.
All retained artifacts and dependencies passed hashes. Overlay and comparison
images were inspected: the shape is coherent, but resemblance is unapproved,
glasses/expressions remain uncertain, and the original nonfrontal camera roll
was wrong. The renderer now constructs an explicit Y-up frame; the old scene
and exact original worker bytes remain preserved.

First failed trial wall time was 326.453s, before rendering.
Accepted CPU run wall time was 372.859s, including 24.078s in the Blender CUDA
stage. Numerical-fit wall time totals 334.515s, about 90% of the accepted run.
These are real command/stage measurements, not estimated GPU speedups.

An initial GPU L-BFGS experiment was rejected: the 53-view generic stage took
20.234s instead of 91.047s but worsened the objective by about 30%.
The accepted Schur/LM benchmark on frozen observations took **8.390s** (10.85x
faster), with objective 0.000973765 versus CPU 0.000975214.
Its retained report is `data\arka\facefitting\benchmarks\cuda-schur-a15979ce.json`.

Fresh CUDA job `20261116T153125Z-44390de9` repeated dense inference, candidate
comparison, all-view fitting and rendering. Outer CLI wall time was **42.187s**
versus **372.859s** CPU: `372.859 / 42.187 = 8.838x`, an 88.686% reduction.
Persisted pipeline-only time was 40.922s; do not compare that with CPU outer time.
The 96-mode candidate's category-balanced held-out error fell from 0.02770004
to 0.02470406 (10.816% improvement); frontal error also improved.
Final all-view median was 0.02210726, not a held-out score. Its final refit took
0.203s; peak process CUDA allocation was 413,934,592 bytes, not total VRAM use.
Both jobs passed artifact/model/source integrity checks. The new comparison
renders were visually inspected: front/45/90-degree views are now upright and
the mesh is coherent, but clay rendering does not establish likeness.
Input references/hashes, eligible/excluded face IDs and the entire training/
held-out split match exactly. The retained comparison record is
`data\arka\facefitting\benchmarks\full-pipeline-cpu-cuda.json`.

See the [researched MetaHuman route and remaining gates](../generation/metahuman-workflow-investigation.md).

## Validation and development

```powershell
python -m unittest ailias.facefitting.tests.test_fitting ailias.facefitting.tests.test_observations -v
$env:AILIAS_CUDA_FITTING_TESTS = "1"
python -m unittest discover -s src\integration_tests\facefitting -p test_cuda.py -v
```

Units use synthetic geometry/fake observations and never import Torch or probe
hardware. The separate explicit CUDA tests compare projection with the CPU
reference, check gradients and recover known identity/opposing expressions.
Actual full-photo trials remain separate from units. A likeness sign-off,
silhouette/surface calibration and rig/animation validation remain open.
The complete isolated suite passed **162 tests** after this implementation;
the two explicit real-CUDA tests also passed separately.

## Primary references

- [Official GNM source and weight license](https://github.com/google/GNM/tree/dfd3e4b96bda2f0061c6013a458b435cf3cb519f)
- [Official GNM NumPy implementation](https://github.com/google/GNM/blob/dfd3e4b96bda2f0061c6013a458b435cf3cb519f/gnm/shape/gnm_xnp.py)
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [Unofficial correspondence and its stated limitations](https://gist.github.com/pundirmr/ecf0c0bb5c93fc1a9534083a8b6d255e)
- [SciPy least squares and numerical Jacobians](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
- [Official PyTorch wheel index used for the matching runtime](https://download.pytorch.org/whl/cu126)
- [PyTorch forward-mode automatic differentiation](https://docs.pytorch.org/docs/stable/generated/torch.func.jacfwd.html)

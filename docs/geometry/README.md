[Back to REPO_ROOT](../../README.md)

# Facial geometry comparison

**Identity resemblance and facial geometry are different signals.** This module
reports geometric errors, not a learned identity or an invented likeness percentage.
See the [research comparison](../imageprocessing/research.md).

```powershell
python -m pip install -e ".[geometry]"
python -m ailias geometry compare arka C:\measurements\reference.json C:\measurements\candidate.json
python -m ailias geometry compare arka C:\measurements\reference.json C:\measurements\candidate.json --fit similarity
```

All input snapshots and reports are saved under `data\arka\geometry\COMPARISON`.
No model downloads, GPU execution or external services are used by this command.

## Input contract

Both files need the **same named, corresponding landmarks**, including
`left_eye` and `right_eye` for interocular normalization. Use consistent eye
landmark definitions, neutral expressions and coordinate conventions.
Coordinates below are illustrative, not measurements:

```json
{
  "dimension": 3,
  "units": "mm",
  "landmarks": {
    "left_eye": [-30, 0, 0],
    "right_eye": [30, 0, 0],
    "nose": [0, -20, 20],
    "chin": [0, -65, -5],
    "forehead": [0, 35, -8]
  },
  "regions": {"eyes": ["left_eye", "right_eye"], "nose": ["nose"]}
}
```

3D requires at least four noncollinear corresponding landmarks; units `mm`,
`cm` or `m` are converted to mm. 2D requires at least three landmarks and matching
`pixels` or `normalized` units. Neither input can contain NaN/Inf or degenerate
landmark geometry. Regions use the reference file's landmark definitions.

Optionally include `surface_points: [[x,y,z], ...]` in **both 3D files**. These
are sampled surface points in the same frame/units as their own landmarks.
Limit: 10,000 points per input. Use the same anatomical crop and reasonably
uniform, comparable sampling. This is not a full-mesh loader.

## Outputs

| Field | Meaning |
|---|---|
| `landmark_error` | Mean, median, RMSE, p95 and max after alignment |
| `normalized_rmse` | Landmark RMSE / reference interocular distance, lower is better |
| `per_landmark`, `regions` | Errors by named point and region after one global fit |
| `candidate_to_reference` | Explicit no-reflection transform and scale |
| `sampled_surface` | Optional bidirectional nearest-sample statistics, unsquared symmetric Chamfer, Hausdorff max/p95 |

`rigid` removes translation/rotation but preserves physical size. `similarity`
also estimates uniform scale: useful when scale is genuinely unknown, but can
hide a wrong head size. Do not select the better-looking score without reporting
which fit was used.

2D scores are only interpretable across matched viewpoints, expression and camera
projection. Five detected photo points cannot establish 3D nose depth, jaw, skull
or ear likeness. Surface sample distances are **not** triangle-surface distance,
NoW benchmark scores, or an animation-quality test.

[Developer guide](dev.readme.md)

[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Geometry comparison

Compare corresponding 2D/3D landmarks and optional sampled surfaces.
This module measures supplied geometry; it does not reconstruct, sculpt or
recognize a face, and does not produce a likeness percentage.

## Usage

```powershell
python -m ailias geometry compare subject C:\measurements\reference.json C:\measurements\candidate.json
python -m ailias geometry compare subject C:\measurements\reference.json C:\measurements\candidate.json --fit similarity
```

Inputs/report snapshots live under `data\ALIAS\geometry\COMPARISON`.
NumPy is supplied by the optional `geometry` extra; no model, GPU or service is required.

## Configuration

There is **no module JSON policy**. Global `--data-root` selects storage;
`geometry compare --fit` selects alignment. Measurement units/regions belong in
the input JSON, not environment variables.

| Setting | Default / allowed values | Meaning |
|---|---|---|
| `--fit` | `rigid`; optional `similarity` | Remove rotation/translation; similarity also estimates uniform scale |
| `dimension` | Required: `2` or `3` | Coordinate dimensionality |
| `units` | 3D: `mm`, `cm`, `m`; 2D: `pixels`, `normalized` | 3D values convert to mm; 2D inputs must use matching units |
| `landmarks` | Required named coordinates | Same names/correspondences in both inputs, including `left_eye` and `right_eye` |
| `regions` | Optional reference-side mapping | Named landmark groups; errors reuse the global fit |
| `surface_points` | Optional in both 3D inputs | Same anatomical crop and reasonably comparable sampling |

At least three noncollinear 2D or four noncollinear 3D landmarks are required.
Coordinates must be finite. Surface samples are capped at 10,000 per input;
this is an implementation bound, not a configurable sampling policy.

## Input example

Illustrative coordinates, not facial measurements:

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

Raw OBJ files or unrelated landmark orders cannot be passed directly.
Extract anatomically corresponding points with consistent definitions first.

## Outputs

| Report field | Meaning |
|---|---|
| `landmark_error` | Aligned mean, median, RMSE, p95 and maximum |
| `normalized_rmse` | RMSE divided by reference interocular distance |
| `per_landmark`, `regions` | Named/region errors under one global alignment |
| `candidate_to_reference` | No-reflection transform and scale |
| `sampled_surface` | Optional bidirectional nearest-sample distances, unsquared symmetric Chamfer and Hausdorff max/p95 |

`rigid` preserves size; `similarity` can hide incorrect head scale.
2D comparison requires matched viewpoint/expression/projection. Sample distances
are not point-to-triangle errors or an animation-quality test.

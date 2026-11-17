[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Geometry internals

## Implementation

| Component | Responsibility |
|---|---|
| `metrics.py` | Pure NumPy alignment, landmark errors and nearest-sample distances |
| `comparison.py` | Persist input/report snapshots; separate identity cosine from five-point geometry |
| `cli.py` | Explicit compare command and fit selection |

`compare_geometry(reference, candidate, fit="rigid")` accepts JSON-like inputs
and returns JSON-safe numeric results. NumPy is imported lazily.

Alignment uses least-squares SVD rotation with determinant +1, translation and
optional positive uniform scale. Reflections/nonrigid warps are forbidden.
Regional errors reuse the global transform rather than hiding offsets by
independently fitting each feature.

Nearest-sample distances use 128-point chunks with a 10,000-point input cap.
Work remains quadratic; replacing this with spatial indexing or triangle
distances would be an explicit metric implementation change.

## Configuration

No JSON/environment policy is implemented. The [usage contract](README.md#configuration)
defines `--fit`, units, correspondence names and optional regions/surface samples.
Normalization uses the reference eye distance; it is not a calibrated likeness threshold.

## Research decisions in plain language

| Decision | Reason / boundary |
|---|---|
| Separate identity and geometry | Recognition cosine cannot measure nose depth or ear placement |
| Preserve size by default | Similarity alignment can conceal wrong scale |
| One global fit for all regions | Prevent local fits from removing actual feature-position errors |
| Report sample distances explicitly | Nearest sampled points are not nearest mesh triangles |
| Require supplied correspondences | Automatic anatomical measurement extraction is not implemented |

Metric scale requires independent calibration; a model's declared unit alone
does not establish real facial dimensions.
See [measurement research](../generation/selection-research.md) and
[metric discussion](../imageprocessing/research.md#what-facial-similarity-can-mean).

## Tests

```powershell
python -m unittest discover -s src\ailias\geometry\tests -t src -v
```

Fixtures cover known transforms, changed nose geometry, unit conversion,
reflection rejection, rigid/similarity behavior, sampled distances and invalid
inputs. These verify mathematics, not a real reconstruction's anatomical accuracy.

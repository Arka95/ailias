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

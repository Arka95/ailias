[Back to REPO_ROOT](../../README.md) | [Usage](README.md)

# Generation internals

| Module | Responsibility |
|---|---|
| `settings.py` | Validated workload/quality policy and workspace tool root |
| `tools.py` | Direct executable calls, process-local DLL paths, timeouts, bounded stderr/stdout logs and installed-version probes |
| `inputs.py` | Current human-label/source integrity, quality filtering, canonical hardlink views and ellipse masks |
| `matching.py` | Read-only native SQLite view-graph measurements; deterministic connection closure |
| `cameras.py` | Hamilton world-to-camera poses, crop-offset intrinsics and actual observation reprojection errors |
| `mesh.py` | COLMAP ASCII textured-PLY conversion preserving per-corner UV seams |
| `pipeline.py` | Independent experimental jobs, gates, native stages, commit-after-validation and job-local finalizers |
| `blender_worker.py` | Blender 4.5 bundled-Python cleanup, coordinate-preserving OBJ export and GPU review renders |
| `unreal.py`, `unreal_worker.py` | UE 5.8 prerequisites, static-mesh import, existing-Identity conform, explicit cloud requests and saved-package verification |

Python orchestrates native C++/CUDA tools and Blender/Unreal's native Python APIs;
it does not reimplement SfM/stereo or introduce a Torch/CUDA environment.
Native COLMAP's batch launcher is bypassed safely using its actual `bin\colmap.exe`
with the launcher's DLL/plugin paths scoped to that child process.

Input view identity includes the curation fingerprint, portrait/source hashes,
crop/mask policy and chosen photos. Reuse never reclassifies a previously rejected
portrait as accepted. Changed/deleted reviewed positives fail closed until explicit
recapture of human decisions. Native views preserve provenance but are not model
inference replay caches: an explicit reconstruction retry performs fresh work.

Manifest statuses distinguish `running`, `blocked`, `failed`, `interrupted` and
`complete`. A complete native reconstruction job means geometry/cleanup products
passed the configured software gates, **not** that the subject, metric cameras,
head completeness, texture albedo, rig or likeness is verified. Manual session
capture/attestation state remains untouched. `metahuman.status` is independent and
distinguishes blocked, mesh-imported, Identity-conformed and cloud-request-returned
states; none means assembled or animation-validated.

Only saved outputs are committed as artifacts. Missing render/texture/native
files fail rather than committing a placeholder. External Unreal `.uasset` and
optional `.ubulk`/`.uptnl`/`.uexp` files are recorded by hash. Editor operations are
not filesystem transactions; partial project writes are reported, not deleted.
Existing destination namespaces and successful operation receipts are not
overwritten. Hard kills can leave a writer lock; follow the shared recovery guide.
Unreal child-worker lifetime/timeout cleanup still requires validation against an
installed engine before production use.

## Configuration and gates

`config\generation.json` is the policy source; relative `tools_root` resolves
against that config file. Defaults bound input count (32), threads (4), GPU index
(0), per-command time (1,200 seconds), feature side (2,000), dense side (1,200),
Poisson depth (9), trim (5), and render size/samples (768/16).

At least five images and 60% of inputs must form one verified-match component
before mapping. The best recovered model must register the same minimum/fraction,
have at least 100 sparse points and mean measured reprojection error at most
2.5 source pixels. Fusion requires 1,000 foreground samples; meshing/cleanup
requires 500 faces. These are experimental viability heuristics, not calibrated
head-accuracy criteria. The separate `generation-poc.json` profile explicitly
relaxes those gates for first visual inspection, not to label fragments a
successful head reconstruction. Mapper initialization and stereo/fusion support
are recorded policy fields; their defaults match the pinned tool's native defaults.

Poisson's density trim is explicit: the native default of 10 discarded too much
of a low-resolution synthetic eight-view capture. Trim 5 is retained in provenance;
Poisson can still infer unsupported/closed regions, so unseen anatomy is not
certified. Raw observations/cameras remain available for independent inspection.

COLMAP camera model is deliberately `SIMPLE_PINHOLE`, one camera per crop.
Portraits contain no usable EXIF focal metadata. Intrinsics and SfM scale are
estimated, not physical calibration. The camera exporter supports only pinhole
models and refuses unhandled distortion. Blender's reference camera converts
OpenCV axes to Blender axes and preserves reference up during orbiting. Texture
atlases contain photographed lighting; they are not extracted intrinsic albedo.

## Tests

Functional units use fake native tools/UE objects, synthetic images/geometry and
temporary directories; no network, hardware, editor or model work runs:

```powershell
& C:\projectWorkspace\.venv\Scripts\python.exe -m unittest ailias.generation.tests.test_generation ailias.generation.tests.test_unreal -q
& C:\projectWorkspace\.venv\Scripts\python.exe -m unittest discover -s src\ailias -t src -q
```

Explicitly authorized hardware checks live outside unit discovery and installed
packages. They generate synthetic objects, **not human heads**, and remove their
temporary photo/mesh/render files. Reports go to the requested path:

```powershell
python src\integration_tests\generation\blender_smoke.py --report C:\diagnostics\blender.json
python src\integration_tests\generation\native_smoke.py --report C:\diagnostics\native-generation.json
```

The native check also accepts `--preview C:\diagnostics\synthetic-preview.png`
to retain a labeled source/reconstructed-camera/orbit comparison outside the
temporary fixture. It checks all three renders are non-flat and cleanup has
zero errors. The preview is synthetic diagnostic evidence, not a likeness test.

## Validation

Verified on November 16, 2026:

| Check | Evidence |
|---|---|
| Portable installation | Official archive SHA256s verified; COLMAP 4.2.1 `bd1fcf6` with CUDA and Blender 4.5.15 `5afeabba6097`; downloaded archives removed |
| Real-photo first benchmark | 24 explicitly kept/quality-eligible `face-001` portraits; 276 possible pairs, one verified pair with 50 inliers, largest component 2; no valid sparse head |
| Gated real-photo replay | Job `20261116T114830Z-26a2f72c`; GPU extractor/matcher creation confirmed in native logs; 3.219 seconds for native stages; blocked before mapping; 7,995,392 transient bytes removed with zero cleanup errors |
| Broader real-photo benchmark | Job `20261116T120330Z-45950e2f`: all 31 quality-eligible human-kept group portraits; 465 possible pairs, 6 verified, largest component 4 (two components of 4), maximum 109 pair inliers; blocked before mapping; 3.516 native seconds; 9,306,112 transient bytes removed, zero cleanup errors |
| First permissive real-photo output | Job `20261116T122510Z-61cf8504`: relaxed initialization registered 3 photos/26 sparse points; 52 fused points became 3,104 faces in 3 fragments, visually inspected and opened in installed Blender 5.3; not a head |
| Less-filtered real-photo PoC | Job `20261116T122845Z-ed50db97`: one supporting stereo source and two-pixel fusion; 3 cameras/27 sparse points, 0.9148 px measured reprojection error, 4,263 fused points and 6,164 faces in 13 disconnected components; visibly incomplete, 90-degree view mostly empty; 44.469 native seconds; 51,923,277 transient bytes removed, zero cleanup errors; retained hashes verified before manual editing |
| Reuse | Both 24-photo attempts reference the identical native input view; no source/portrait pixel copies |
| Actual Blender CUDA integration | Textured six-vertex/eight-face synthetic octahedron; unchanged exported vertex positions/topology, preserved camera-up orientation, three non-flat PNGs and a `.blend`/OBJ/material produced using RTX 4070 SUPER; 3.172 seconds on the recorded run; fixture files removed |
| Full actual native integration | Synthetic eight-view textured sphere; GPU extraction/matching and CUDA stereo through meshing/texturing/Blender CUDA; all 8 cameras, 977 sparse points, mean 0.2983 px reprojection error, 12,722 fused points, 67,241 faces; three non-flat renders and hashed outputs; 56.031 native seconds; 58,009,622 transient bytes removed with zero errors, temporary fixture removed |
| Visual diagnostic | Retained synthetic comparison inspected: partial open surface, texture brightening under new lighting and unsupported profile coverage remain visible; execution success is not watertightness, measured albedo or photometric fidelity |
| Isolated regression | 111 functional units, including saved-external-package verification and receipt-without-asset failure; no model, hardware, editor or network execution in discovery |
| PoC policy regression | 26 focused generation/Unreal units passed, including explicit native option forwarding, policy provenance, unchanged non-production/likeness flags and invalid-angle/support rejection |
| Unreal | No registered/common-path engine was found; `unreal-doctor`/import correctly blocked without launching, collecting credentials, downloading engine content or making placeholder assets |

The real-photo result applies to that selected set, not every possible subject,
view combination or face-reconstruction model. Sparse correspondences are not a
face-recognition score. The source plan's real head-to-MetaHuman milestone remains
open. No realistic face reconstruction, full-head accuracy, MetaHuman assembly,
expression/rig validation, or final likeness sign-off is claimed.

### Primary references

- [COLMAP 4.2.1 distribution](https://github.com/colmap/colmap/releases/tag/4.2.1);
  archive digests are pinned in the tool manifest.
- [COLMAP command line](https://colmap.github.io/cli.html) and
  [camera/mesh formats](https://colmap.github.io/format.html).
- [Pinned COLMAP texturing implementation](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/exe/mvs.cc)
  and [PLY writer](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/util/ply.cc):
  per-face texture coordinates must not be silently lost by Blender's PLY importer.
- [Official Blender 4.5.15 checksums](https://download.blender.org/release/Blender4.5/blender-4.5.15.sha256)
  and [Blender 4.5 operators](https://docs.blender.org/api/4.5/bpy.ops.wm.html);
  actual operator signatures and execution were checked locally.
- [Epic's UE 5.8 MetaHuman Python workflow](https://dev.epicgames.com/documentation/metahuman/metahuman-creator-python-scripting-in-unreal-engine?application_version=5.8)
  and [From Mesh](https://dev.epicgames.com/documentation/metahuman/from-mesh).
  Local engine/API execution is not inferred from documentation availability.

3DDFA_V2 was considered but not downloaded/adopted: its official repository's
[modified Basel model statement](https://github.com/cleardusk/3DDFA_V2/blob/1b6c67601abffc1e9f248b291708aef0e43b55ae/bfm/readme.md)
limits that asset to academic use, separately from its MIT code. No license/access
or production suitability was assumed, and no Torch/ONNX/CUDA stack was installed.

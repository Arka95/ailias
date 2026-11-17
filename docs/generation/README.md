[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Native generation

An alternative photogrammetry path: curated masked photographs -> COLMAP
reconstruction/texturing -> minimal Blender cleanup -> optional static Unreal import.
It is not the complete-head prior fitter; use [facefitting](../facefitting/README.md) for that.

## Inputs and outputs

| Stage | Required input | Retained resources |
|---|---|---|
| Input preparation | Completed extraction run/group, current curation and cleansing | Hardlinked native RGB, aligned masks, selection audit/contact sheets |
| Reconstruction | Prepared views with a viable verified-match graph | Sparse model, estimated cameras, fused samples, textured OBJ/material/atlas |
| Cleanup | Retained reconstruction products | Coordinate-preserving OBJ, `.blend` and diagnostic renders |
| Unreal import | Cleaned mesh/material/texture, compatible project | Static-mesh packages and hashes |
| Legacy Identity conform | Already-solved Identity plus Character | Native conform receipt/packages; optional explicit cloud requests |

Jobs live in `data\ALIAS\generation\runs`; reusable native views live in
`generation\inputs`. Real-photo trials currently yield blocked jobs or fragments,
not a validated complete head.

## Usage

```powershell
.\scripts\setup-generation-tools.ps1
python -m ailias generation doctor
python -m ailias generation select-inputs subject RUN_ID face-001
python -m ailias generation reconstruct subject RUN_ID face-001
python -m ailias generation status subject JOB_ID
python -m ailias generation clean-mesh subject JOB_ID
```

`reconstruct` refreshes configured cleansing; `select-inputs` only audits existing
derivatives. Reconstruction does fresh native work, not inference replay.
The cleanup retry requires retained reconstruction products.

## Configuration

`--generation-config FILE` selects `config\generation.json`.
Relative `tools_root` resolves against that file, not the current directory.
The table gives the effective standard profile, including omitted dataclass defaults.

| Key under `policy` | Default | Meaning |
|---|---|---|
| `max_images` | `64` | Maximum selected distinct photos |
| `selection_expand_groups` | `true` | Associate compatible groups around the seed-subject hypothesis |
| `selection_allow_quality_warnings` | `false` | Allow selected soft quality warnings; never bypass identity/curation checks |
| `use_cleansed_portraits` | `true` | Use current native-grid head RGB/alpha; direct dataclass default is `false` |
| `min_registered_images` | `5` | Minimum connected/registered image count |
| `min_registered_fraction` | `0.6` | Minimum connected/registered share of selected inputs |
| `min_sparse_points` | `100` | Minimum recovered sparse point count |
| `max_reprojection_error_px` | `2.5` | Maximum mean measured source-pixel reprojection error |
| `mapper_init_min_inliers` | `100` | Inlier support for mapper initialization |
| `mapper_init_min_tri_angle` | `16` | Minimum initialization triangulation angle, degrees |
| `mapper_abs_pose_min_inliers` | `30` | Inlier support for subsequent camera registration |
| `stereo_min_consistent` | `2` | Minimum consistent source support for stereo |
| `fusion_min_pixels` | `5` | Minimum supporting pixels for fused samples |
| `max_feature_image_size` | `2000` | Maximum feature-extraction image side, pixels |
| `max_dense_image_size` | `1200` | Maximum dense-stereo image side, pixels |
| `min_fused_points` | `1000` | Minimum foreground fused samples |
| `min_mesh_faces` | `500` | Minimum mesh/cleanup face count |
| `cpu_threads` | `4` | Native CPU thread budget |
| `gpu_index` | `0` | Explicit COLMAP GPU index |
| `timeout_seconds` | `1200` | Per native-command timeout |
| `poisson_depth` | `9` | Poisson reconstruction octree depth |
| `poisson_trim` | `5` | Density-based surface trimming threshold |
| `render_size` | `768` | Blender diagnostic image side, pixels |
| `render_samples` | `16` | Blender diagnostic sample count |
| `blender_device` | `CUDA` | `CUDA`, `OPTIX` or explicit `CPU`; no silent device switch |

Top-level `tools_root` defaults to `..\..\tools`.
`config\generation-tools.json` pins each native tool's `version`, `url`,
`sha256`, `extract_to` directory and `executable`. Setup uses these specifications;
jobs do not download tools.

### Experimental profile

Select `config\generation-poc.json` explicitly for fragment inspection.

| Parameter | Standard -> PoC |
|---|---|
| `selection_allow_quality_warnings` | `false` -> `true` |
| `min_registered_images`, `min_registered_fraction` | `5 / 0.6` -> `3 / 0.04` |
| `min_sparse_points`, `max_reprojection_error_px` | `100 / 2.5` -> `20 / 8` |
| `mapper_init_min_inliers`, `mapper_init_min_tri_angle` | `100 / 16` -> `20 / 0.5` |
| `mapper_abs_pose_min_inliers` | `30` -> `15` |
| `stereo_min_consistent`, `fusion_min_pixels` | `2 / 5` -> `1 / 2` |
| `min_fused_points`, `min_mesh_faces` | `1000 / 500` -> `50 / 50` |
| `poisson_depth`, `poisson_trim`, `render_samples` | `9 / 5 / 16` -> `10 / 2 / 8` |

These are viability gates, not anatomical accuracy thresholds. More triangles
do not turn disconnected fragments into a complete head.

## Unreal: explicit installation and manual Identity solve

This heading describes the **legacy adapter**, not a required manual step in
the personal-head automation path. The current head-only Character route is in
[facefitting](../facefitting/README.md).

```powershell
$ue = 'C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
$project = 'C:\path\to\Review.uproject'
python -m ailias generation unreal-doctor --editor $ue --project $project
python -m ailias generation unreal-import subject JOB_ID --editor $ue --project $project
python -m ailias generation unreal-conform subject JOB_ID --editor $ue --project $project --identity /Game/Characters/Identity --character /Game/Characters/Character
```

The existing-Identity adapter requires a solved Identity and compatible Character.
Only `--cloud-services` requests cloud rig/texture services. It does not package
automatic likeness refinement or runtime assembly.
`unreal-doctor` checks editor/project/plugins and reports Core Data separately;
readiness is not a completed solve.

## Limits and cleanup

Estimated `SIMPLE_PINHOLE` cameras and SfM scale are not metric calibration.
Photo atlases retain illumination. Blender removes degenerates/loose vertices
and repairs normals without filling anatomy, smoothing features or rescaling the head.

Finalizers remove databases, undistortion/depth caches and owned partial outputs,
retaining validated products and diagnostics. Native import may leave partial
packages after failure; the tool reports them rather than deleting external project assets.
See [validation](dev.readme.md#validation) and [cleanup safeguards](../dev.readme.md#resource-cleanup).

[Back to REPO_ROOT](../README.md)

# Manual-first workflow

For an uncurated photo directory, first use the optional
[imageprocessing module](imageprocessing/README.md) to create and review alias
groups, crops and masks. That module does not automatically attest this
workflow's capture conditions.

All commands output JSON on stdout; errors go to stderr. Exit codes: `0` operation
succeeded, `1` invalid input/I/O/configuration/locked session, `2` blocked validation,
failed individual image intake or an integrity problem on `status`. A successful
`intake` does not mean photos passed validation.

Global options precede the command:

```powershell
python -m ailias --data-root C:\ailias-data --config C:\projectWorkspace\ailias\config\settings.json status first-face
```

Configuration precedence: CLI `--data-root` overrides the JSON `data_root`;
relative JSON paths resolve against the configuration file's directory. Other
JSON fields control quality and coverage policy. Unknown/invalid fields fail.

## 1. Capture and intake

Capture 8-15 high-resolution photos with diffuse lighting, fixed focal length,
neutral expression, no glasses, and forehead/ears visible. Include front, left
and right 30/45/90 degrees, slightly above/below, and facial-feature close-ups.
The implemented angular gate requires **nine distinct views**, so nine is the
effective minimum for a complete gate with the default view set.

`intake SESSION FOLDER` scans only that folder, accepts JPG/PNG/WebP/TIFF, copies
files by SHA256 and reports corrupt/unreadable files without losing valid intake.
Identical bytes deduplicate. Re-intake is additive; curate the input before intake
or use a new session to remove unwanted photos. At most 15 **accepted** images
may pass; rejected images remain auditable.

Checks include EXIF orientation, minimum short side, whole-image edge energy and
dark/bright pixel fractions. Blur and exposure checks are intentionally
uncalibrated heuristics: detailed backgrounds can hide facial blur. They are not
face detection, expression detection, landmark estimation or a likeness score.

## 2. Human review and coverage

```powershell
python -m ailias review first-face PHOTO_ID --yaw -45 --pitch 0 --neutral --unobstructed --consistent
python -m ailias review first-face PHOTO_ID --yaw 0 --pitch 0 --no-neutral --unobstructed --consistent
python -m ailias validate first-face
```

Angles are **manually supplied**: yaw negative = camera on subject's left;
positive = right; pitch positive = camera above subject. Eye-level target yaw
values: `0`, `-30`, `30`, `-45`, `45`, `-90`, `90`; above/below targets:
`yaw=0,pitch=20/-20`. Default tolerance is 10 degrees per axis. Each photo
contributes at most one nearest target, not multiple coverage slots.

`--consistent` attests lighting and focal-length consistency against the set.
Negative flags persist rejection reasons. Optional `--landmarks FILE.json`
persists manually supplied normalized coordinates on the EXIF-corrected image:

```json
{"nose_tip": [0.5, 0.6], "left_eye": [0.35, 0.4]}
```

The `review` command runs no automatic detector. Optional preprocessing's five
detected landmarks are retained in its own run manifest, not imported as a human
capture review. Landmarks/close-up coverage are not yet a gate.
The validator requests missing yaw/pitch angles but does not estimate
information gain, camera distance or regional reconstruction confidence.

## 3. Reconstruct outside ailias

Benchmark COLMAP, Meshroom or a suitable face-reconstruction model externally.
No backend is selected yet; do not treat a generic textured reconstruction as a
validated facial reconstruction. Preserve mesh, texture, recovered camera poses,
reprojection errors and tool settings/runtime/VRAM measurements.

Attach a non-empty OBJ/PLY/FBX mesh, readable texture and `cameras.json`:

```powershell
python -m ailias attach first-face reconstruction --mesh C:\results\head.obj --texture C:\results\albedo.png --cameras C:\results\cameras.json --note "Manual reconstruction; record tool/version and observations here"
```

`cameras.json` must map **every accepted full photo SHA256** to:

```json
{
  "ACCEPTED_FULL_PHOTO_SHA256": {
    "world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    "intrinsics": [[1200,0,1024],[0,1200,1024],[0,0,1]],
    "reprojection_error_px": 0.8
  }
}
```

Values above are illustrative, not recovered poses. Convention for imported
records: OpenCV camera axes (x right, y down, z forward); world units meters;
intrinsics in pixels for the EXIF-corrected source resolution. Convert tool
outputs to this convention first. Shape/finite-number validation is implemented;
physical camera calibration, matrix inversion and reprojection are not.

## 4. Clean and conform manually

In Blender remove background/floating geometry, fill holes, orient/scale the head,
and transfer textures. Inspect watertightness and save a clean OBJ/FBX. These
checks are human responsibilities, not certified by `attach`.

```powershell
python -m ailias attach first-face cleanup --mesh C:\results\clean.obj --texture C:\results\clean-albedo.png --note "Manually inspected geometry, scale, holes and texture"
python -m ailias attach first-face metahuman --evidence C:\results\editor.png --note "Manual UE Mesh to MetaHuman result; include UE version and asset path"
```

The MetaHuman handoff is a screenshot plus human attestation, **not an imported
Unreal project or a verified rig**. Retain the actual Unreal project and textures
in your normal backup system. Target UE 5.8+ per the source plan; consult the
official version-specific workflow linked in [scope](analysis.md).

## 5. Record a visual baseline

Manually render neutral front, 45-degree and 90-degree views, using the recovered
cameras when available. Choose accepted reference photo IDs with matching reviewed
angles. ailias checks reference angles, but cannot verify the render camera yet.

```powershell
python -m ailias baseline first-face `
  --front-reference FRONT_ID --front-render C:\results\front.png `
  --three-quarter-reference ANGLED_ID --three-quarter-render C:\results\angle.png `
  --profile-reference PROFILE_ID --profile-render C:\results\profile.png `
  --note "Human observations: nose projection, jaw, ears and remaining mismatch"
python -m ailias status first-face
```

The `comparison` artifact is a three-row reference/render contact sheet.
`baseline_recorded=true` means manual receipts and artifacts are present, **not**
that likeness, animation, geometry quality or a production MetaHuman is verified.
`production_ready` stays false in this increment.

## Resume, revisions and recovery

Each successful command saves an atomic manifest. Unchanged intake/validation
leaves later handoffs intact. Changed photos, review or policy invalidate dependent
stages; previous records and content-addressed artifacts remain in history.
Re-run validation before attaching stages after policy changes. `status` hashes
all tracked artifacts; it reports corruption rather than claiming completion.

Failed handoffs leave the last committed stage intact. Partial copies may remain
unreferenced; no automatic deletion/garbage collection runs. One writer lock per
session prevents concurrent updates. After an interrupted process, inspect the PID
in `data\SESSION\.lock` and remove **only that lock** once its writer is confirmed
stopped. Never remove another active session's lock. Back up `data` explicitly;
Git ignores it. No rollback command or automatic external-tool restart exists yet.

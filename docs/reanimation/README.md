[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md) | [Research and decisions](implementation.md)

# Reanimation: existing MetaHuman to portrait reel

This separate module consumes an **already assembled, rigged MetaHuman Blueprint**
and narration. It does not create, repair or certify facial likeness. Earlier
static heads and unassembled Character drafts are not suitable inputs.

The adapter implements offline MetaHuman Animator full-face lip sync, generated
blinks/head motion, an excited mood, a portrait camera, soft daylight-style
lighting, a chroma-key green backdrop and Movie Render Queue PNG output. FFmpeg
then encodes H.264/AAC at 1080x1920, 30 fps. **The complete Taro narration reel
passes native rendering, encoding, full decode and duration checks.**
Subjective lip-sync and delivery review remains with the owner.
Mood is not a replacement for an energetic
voice recording, and this version does not generate arm or torso gestures.

## Commands

Use the shared workspace environment. Supply a separate UE 5.8 project containing
your rig; the command never enables plugins or modifies project settings silently.
Enable `PythonScriptPlugin`, `EditorScriptingUtilities`, `MetaHuman`,
`MovieRenderPipeline` and `SequencerScripting`.

```powershell
Set-Location C:\projectWorkspace\ailias
$ue = 'C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
$project = 'C:\path\to\your\ExistingMetaHuman.uproject'
$ffmpeg = 'C:\projectWorkspace\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe'
python -m ailias reanimation doctor arka --editor $ue --project $project --ffmpeg $ffmpeg
python -m ailias reanimation run arka --audio C:\path\to\narration.wav --editor $ue --project $project --blueprint /Game/MetaHumans/Arka/BP_Arka --ffmpeg $ffmpeg
python -m ailias reanimation status arka JOB_ID
```

`BP_Arka` above is an **example**, not an asset found in this workspace. The
Blueprint must have a compiled class, one `Face` skeletal component, a `head`
bone and a RigLogic post-process animation Blueprint.

Omit `--blueprint` and `--ffmpeg` to solve audio only. That produces a saved
Performance asset with status `audio_solved_awaiting_rigged_metahuman`; it does
**not** produce or claim a video. Audio must be nonsilent mono PCM16 WAV,
16-96 kHz, at most 120 seconds. Voice generation happens upstream through Yapit
MCP; this module accepts its WAV without replacing the selected voice/provider.

## Camera, light and GPU policy

Override `config\reanimation.json` with `--reel-config`. Defaults use an 85 mm
lens, 20.25x36 mm portrait filmback and a 52 cm tall field of view, placing the
camera about 123 cm from the head target. `front_axis` must match the imported
rig's neutral facing direction; the initial setting is `+Y`. Head motion can
change the crop, so inspect the actual human render before publishing.

The unlit background defaults to saturated chroma green, `#00FF00`. Two
200x200 cm, 5600 K lights use a 900-lumen key and 65% fill. Manual exposure is
-5.5 EV; change `exposure_compensation_ev`, `key_lumens`, `fill_ratio` or
`light_size_cm` in a custom policy to adjust the treatment.

Depth of field is disabled to keep the face and keying edges sharp, rather than
focusing behind the eyes at the head joint. Eight temporal samples are used.
The tone curve, bloom, vignette and lens flare are disabled; the unlit backdrop
compensates manual exposure. Actual PNG background samples verify `(0,255,0)`.
H.264 compression can change pixel values; this is not an alpha-channel output.

| Stage | Device policy |
|---|---|
| UE shading/rendering | D3D12 GPU; no NullRHI or CPU rendering fallback |
| Final H.264 | NVIDIA NVENC by default; failures stop publication |
| Installed UE Speech2Face | CPU: Epic's installed native implementation uses `NNERuntimeORTCpu` |
| PocketTTS narration | Record the actual Yapit runtime device; no promise of unsupported CUDA |

CPU encoding requires explicitly selecting `libx264` in a custom policy; the
default never switches silently. No shared Torch/CUDA environment is replaced.

## Results and limitations

The real Yapit MCP generation produced **24.74 seconds** of PocketTTS narration
using the existing `testuser` voice (CPU). It is saved at
`data\arka\reanimation\narration\20261117_pocket_testuser_9c37a2\narration.wav`,
with the script and MCP/provider provenance alongside it. No voice assets or
shared model stacks were changed.

Native job `5a0ae3435a1546f181ee87ee1032a267` processed **742 animation frames**
with `EXCITEMENT`, head motion and blinks. Its saved Performance was independently
reloaded in a separate UE process and confirmed exportable. This is native
animation data, **not an animated human video**.

Each job retains `request.json`, `native-receipt.json`, logs, the exact worker
and a manifest under `data\arka\reanimation\runs\JOB_ID`. Native assets are
created under a new `/Game/Ailias/Reanimation/RJOB_ID` namespace. Existing
characters, scenes and source WAVs are not overwritten.

A video is published only after exact contiguous portrait frames, native success,
H.264/AAC metadata, decoded frame count and audio/video duration checks.
`video_rendered_awaiting_visual_review` means the file passed technical checks,
not that lip sync, likeness, headroom or lighting passed human review.

Current full CLI job **`7f59ccf3576645c2b452dbbdab985d19`** produces
`data\arka\reanimation\runs\7f59ccf3576645c2b452dbbdab985d19\portrait-reel.mp4`.
It contains **743 frames at 1080x1920/30 fps**, with the original PocketTTS/testuser
narration encoded as AAC. Native processing/rendering exited cleanly in
216.015 seconds. Video duration is 24.766667 seconds; decoded audio covers
24.789333 seconds. Both streams decode completely, and all manifest artifact
hashes verify. These are technical coverage checks, not phoneme-sync certification.

**Thirteen isolated units pass.** Earlier technical fixtures crashed with
`0xC0000005`; the startup-script-owned shutdown change now passes on real Taro
renders. No exit-code gate is bypassed. See the [development evidence](dev.readme.md)
for the fixes and retained failures.

The independent test subject selected on the owner's instruction is **Taro**
from Epic's free [MetaHumans sample on Fab](https://www.fab.com/listings/0281d63e-71f7-4e07-a344-5fa721ac4d35).
The downloaded source project identifies itself as UE 5.7. Native migration into
the isolated UE 5.8 review project includes **441 Taro dependency packages** and
no Ada character. Every source package still matches the downloaded cache;
native migration rewrites target package bytes, so target hashes are recorded
separately rather than claimed to be identical copies.

The actual `/Game/MetaHumans/Taro/BP_Taro` Blueprint loads with its Face mesh,
head bone and facial post-process animation Blueprint. Its sampled facial bones
confirm the default `+Y` front direction. These checks are recorded in
`data\arka\reanimation\taro-migration.json` and `taro-rig-audit.json`;
they are not a finished rendered reel. The source project and cache are preserved.
This sample does not depend on fitting the owner's face.

## Optional apparel: planned

Apparel selection is a deferred, opt-in feature and does not block this reel.
When enabled, the requested default is a dark-green crew-neck T-shirt, with
overridable clothing assets/style/color. The current video retains Taro's
original hoodie; no shirt replacement or apparel CLI is implemented yet.
Green garments can conflict with background keying, so wardrobe and backdrop
choices must remain independent.

[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md) | [Research and decisions](implementation.md)

# Reanimation: existing MetaHuman to portrait reel

This separate module consumes an **already assembled, rigged MetaHuman Blueprint**
and narration. It does not create, repair or certify facial likeness. Earlier
static heads and unassembled Character drafts are not suitable inputs.

The adapter implements offline MetaHuman Animator full-face lip sync, generated
blinks/head motion, an excited mood, a portrait camera, soft daylight-style
lighting, a light-green backdrop and Movie Render Queue PNG output. FFmpeg then
encodes H.264/AAC at 1080x1920, 30 fps. **Human rendering remains unvalidated until
a genuine assembled rig is supplied.** Mood is not a replacement for an energetic
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

The light-green plane is unlit. Two broad 5600 K lights provide a soft key and
35% fill. These are starting settings, **not visually approved skin lighting**.
The backdrop is not an alpha channel or chroma-key guarantee.

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

Nine isolated units pass. Real UE camera/light/MRQ-setting checks and a
three-frame 1080x1920 D3D12 render, NVENC encode and audio/video decode check passed.
This fixture is only a green technical test with a synthetic tone, not a human
reel or the requested voice narration. See the [development evidence](dev.readme.md)
for the current native integration result and remaining rig blocker.

[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# Reanimation

Animate an **existing assembled, rigged MetaHuman Blueprint** from supplied
narration. UE solves face/head motion and renders portrait PNGs; FFmpeg encodes
H.264/AAC. This module does not construct a head, repair likeness or synthesize speech.

## Inputs and outputs

| Input | Requirement |
|---|---|
| Narration WAV | Nonsilent mono PCM16, 16-96 kHz, within configured duration |
| UE project/editor | Compatible UE 5.8 project with Python, MetaHuman, Movie Render Pipeline and Sequencer scripting plugins |
| Blueprint | Compiled actor, one Face skeletal component, head bone and RigLogic post-process |
| FFmpeg | Explicit executable with the selected encoder |

Jobs under `data\ALIAS\reanimation\runs` retain the manifest, exact worker,
request/receipt, logs, native assets, verified MP4 and first/last PNGs.
An unrigged Character or static mesh is not a substitute for the Blueprint.

## Usage

```powershell
$ue = 'C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
$project = 'C:\path\to\ExistingMetaHuman.uproject'
$ffmpeg = 'C:\path\to\ffmpeg.exe'
python -m ailias reanimation doctor --alias subject --editor $ue --project $project --ffmpeg $ffmpeg
python -m ailias reanimation run --alias subject --voice C:\audio\narration.wav --blueprint /Game/MetaHumans/Character/BP_Character --editor $ue --project $project --ffmpeg $ffmpeg --resolution 540x960
python -m ailias reanimation status --alias subject JOB_ID
```

The asset path above is an example. Omitting both Blueprint and FFmpeg performs
audio solve only, with status `audio_solved_awaiting_rigged_metahuman`; no video
is produced. `--voice` and compatibility spelling `--audio` select the same WAV,
from any provider. There is no TTS/Yapit runtime or test dependency.

Reanimation and native portrait review reuse the project-owned editor by
default; completion does not close it. [Session commands/configuration](../generation/README.md#reusable-unreal-editor)
control its lifetime. This does not reuse an old speech solve or render:
each requested reel still solves/renders fresh content under its existing checks.

## Configuration

`run --reel-config FILE` selects `config\reanimation.json`.
Per-run overrides win without changing the policy file, source audio or model.

`--reel-config config\character-review.json` selects the independent brighter
character-review exposure profile without modifying the reanimation
preset. It is also used by `facefitting video-create --review-config`.
Character construction/hairstyle changes remain in facefitting; assemble its
saved Character with `unreal-features --assemble` before passing the resulting
Blueprint here. Only explicitly authorized `--cloud-services` requests may
obtain missing native rig/textures; an older head's rig is not substituted.

| CLI option | Meaning |
|---|---|
| `--alias` | Output namespace, independent of voice/model; legacy positional alias is supported |
| `--blueprint` | Compatible existing native character |
| `--voice` / `--audio` | Exact source WAV; never a synthesis profile |
| `--resolution WIDTHxHEIGHT` | Override even 9:16 dimensions; `540x960` for drafts |
| `--character-position X Y Z` | Override world placement in cm |
| `--character-yaw DEGREES` | Override character yaw; camera follows its declared neutral front |
| `--screen-position U V` | Override normalized head framing target; top-left `(0,0)` |
| `--keep-frames` | Retain every rendered PNG instead of first/last only |

### `config\reanimation.json`

| Key | Default | Meaning |
|---|---|---|
| `width`, `height` | `1080`, `1920` | Even 9:16 production dimensions; maximum 2160x3840 |
| `fps` | `30` | Output/sequence rate, 1-60 fps |
| `focal_length_mm` | `85` | Portrait lens focal length |
| `sensor_height_mm` | `36` | Filmback height; width follows aspect ratio |
| `framing_height_cm` | `52` | Vertical world-space field of view at the framing target |
| `head_center_offset_cm` | `6` | Vertical framing offset relative to the measured head |
| `front_axis` | `+Y` | Rig's neutral front: `+X`, `-X`, `+Y`, `-Y` |
| `character_location_cm` | `[0,0,0]` | Actor world translation |
| `character_yaw_degrees` | `0` | Actor yaw, -180 to 180 degrees |
| `subject_screen_position` | `[0.5,0.5]` | Normalized head target in the frame |
| `background_srgb` | `[0,1,0]` | Unlit chroma background; encoded MP4 is not alpha output |
| `exposure_compensation_ev` | `-5.5` | Manual exposure in stops |
| `mood_intensity` | `0.7` | Native excitement intensity, 0-1; does not change the voice recording |
| `key_lumens` | `900` | Key-light intensity |
| `fill_ratio` | `0.65` | Fill intensity as a fraction of the key |
| `light_size_cm` | `200` | Rectangular soft-light size |
| `temperature_kelvin` | `5600` | Light colour temperature |
| `temporal_samples` | `8` | Movie Render Queue temporal samples |
| `warmup_frames` | `32` | Engine warmup budget |
| `encoder` | `h264_nvenc` | Explicit NVIDIA encoder; `libx264` selects CPU, never an automatic fallback |
| `max_audio_seconds` | `120` | Input duration limit, at most 600 seconds |
| `timeout_seconds` | `1800` | Native/encoding operation timeout |

Sensor width = sensor height x width / height. Camera distance =
framing height x focal length / sensor height. Changing resolution at fixed
aspect does not move the camera or lights.

## Validation and limitations

| Gate | Guarantee / boundary |
|---|---|
| Native save/render | Compatible actor and saved Performance; clean editor completion required |
| Frames | Exact contiguous count, portrait size and readable PNGs |
| Encoding | Selected encoder, H.264 video and supplied narration as AAC |
| Media verification | Complete video/audio decode and duration checks |
| Visual quality | Lip sync, acting, facial resemblance and headroom require separate evaluation |

Rendering does not repair incorrect shape, skin, hair or facial resemblance.
No arm/torso gestures or apparel-selection API is implemented.

Rendering uses D3D12/Movie Render Queue; the installed Speech2Face solve is CPU.
Depth of field and tone/lens effects are disabled for sharp face/keying edges.
The default background is green, so wardrobe colour remains a separate concern.

## Cleanup

Finalization removes partial videos and redundant interior PNGs while retaining
diagnostic boundary frames, logs, native assets and verified media.
`--keep-frames` opts out of frame reduction.
See [resource cleanup](../dev.readme.md#resource-cleanup) and
[implementation research](implementation.md).

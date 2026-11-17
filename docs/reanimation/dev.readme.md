[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Reanimation internals

## Workflow and communication

| Component | Responsibility / channel |
|---|---|
| `cli.py`, `settings.py` | Argument/policy validation and derived camera/placement geometry |
| `pipeline.py` | Alias lock, immutable jobs, owned editor/encoder processes and publication |
| `unreal_worker.py` | Request JSON -> native audio solve, stage/sequence, MRQ frames and receipt |
| `media.py` | WAV/frame checks, explicit FFmpeg commands and full media decoding |

Host-to-editor communication uses a request-file path in the child environment.
The editor emits packages, PNGs, logs and JSON receipts. The host checks both
native completion and actual files before encoding. FFmpeg subprocesses encode
and completely decode video/audio before a verified MP4 is retained.

| Phase | Contract |
|---|---|
| Input | Exact external WAV plus independently selected compatible Blueprint |
| Audio-only | Save Performance; do not imply a rendered human |
| Staging | New map/sequence; configured placement without changing facial Control Rig keys |
| MRQ | Asynchronous callbacks retained inside the owned editor process |
| Exit | Startup-script owner controls shutdown; nonzero exit remains failure |
| Encoding | NVENC/AAC by default; no silent encoder fallback |
| Finalization | Retain verified video, boundary frames and evidence; remove redundant scratch |

No TTS adapter, voice-model selection, Yapit import or cloud identity request
is involved. Model/audio/placement/resolution remain independent.

## Camera and lifecycle

Camera distance follows the selected framing/lens geometry. Yaw rotates the
declared front/right axes; normalized screen placement creates a parallel target
shift, preserving perspective. Lights follow the measured head, not the shifted
camera target. Depth of field is disabled.

Native export may add a static reference-pose root transform. The adapter only
normalizes that selected actor's static root channels to configured placement;
animated or ambiguous roots fail rather than losing motion.

MRQ callbacks use Python keep-alive. The worker releases that flag and lets the
engine startup-script owner shut down; directly racing editor quit against
render completion previously caused native shutdown failures.

## Configuration

The [usage guide](README.md#configuration) owns all JSON keys, defaults, units
and CLI precedence. `ReelPolicy` checks even 9:16 dimensions, bounded audio,
finite placement, explicit encoder and lens/light ranges.
Derived framing and effective overrides are recorded in each request/manifest.

## Tests

```powershell
python -m unittest ailias.reanimation.tests.test_reanimation -v
```

Units use temporary WAV/PNG fixtures and fake native/decoding operations.
They cover audio format/duration, aliases, independent placement, equivalent
`--voice`/`--audio`, exact frames, media failures and cleanup. No UE, model,
network, hardware or TTS service is needed.

## Integration evidence

| Retained evidence | Scope |
|---|---|
| Sample-character rig audit/migration records | Compatible native Blueprint, Face/head and post-process loading |
| Sample full-resolution reel | 743 frames, 1080x1920/30 fps, clean native exit, H.264/AAC and complete decode |
| Personal-draft reel | 743 frames, 540x960/30 fps, same narration and technical verification; likeness remains below expectations |
| Failed receipts/logs | Retained alongside diagnostics; not reclassified as successful videos |

Records live under `data\arka\reanimation`; the personal native package lineage
is referenced by `data\arka\facefitting\reviews\personal-runtime-character.json`.
Technical decoding/timing checks do not certify phoneme alignment, acting or identity.
Detailed implementation research is in [implementation.md](implementation.md).

## Operational limits

Jobs are synchronous and serialized per alias, not a queue service.
Crashes can leave partial native assets/running manifests; inspect and create a
fresh revision rather than reusing the namespace. Source WAV hashes are checked
again after animation. Existing projects/assets are not silently upgraded or pruned.

A separate later render currently starts another solve instead of resuming a
saved audio-only Performance. Arm/torso gestures and wardrobe APIs remain outside
the current implementation.

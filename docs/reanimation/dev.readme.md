[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# Reanimation internals

## Workflow and communication

| Component | Responsibility / channel |
|---|---|
| `cli.py`, `settings.py` | Argument/policy validation and derived camera/placement geometry |
| `pipeline.py` | Alias lock, immutable jobs, native/encoder requests and publication |
| `unreal_worker.py` | Request JSON -> native audio solve, stage/sequence, MRQ frames and receipt |
| `media.py` | WAV/frame checks, explicit FFmpeg commands and full media decoding |

Host-to-editor communication uses the project-owned session's request mailbox;
explicit one-shot launches pass the request path in the child environment.
The editor emits packages, PNGs, logs and JSON receipts. The host checks both
native completion and actual files before encoding. FFmpeg subprocesses encode
and completely decode video/audio before a verified MP4 is retained.

| Phase | Contract |
|---|---|
| Input | Exact external WAV plus independently selected compatible Blueprint |
| Audio-only | Save Performance; do not imply a rendered human |
| Staging | New map/sequence; configured placement without changing facial Control Rig keys |
| MRQ | Asynchronous callbacks retained inside the owned editor process |
| Completion | Job receipt ends the operation, not the shared editor; native process failure remains failure |
| Encoding | NVENC/AAC by default; no silent encoder fallback |
| Finalization | Retain verified video, boundary frames and evidence; remove redundant scratch |

No TTS adapter, voice-model selection, Yapit import or cloud identity request
is involved. Model/audio/placement/resolution remain independent.

## Timing

`ReelPipeline` owns request preparation, native execution, native-package
verification, frame validation, encoding, full media verification, source
verification and finalization as separate journaled steps. `create_reel()`
retains its existing Python API.

The native receipt contains `stages.audio_lipsync`,
`stages.place_character_and_export_animation` and `stages.render_video`.
Audio-only and neutral-review jobs omit the stages they do not execute.
The render timer ends in the completion callback after delegate cleanup, not
when asynchronous rendering is launched. Failed rendering or delegate cleanup
records a failed stage before publishing the terminal receipt.

The host `stages.native_execution` contains those native child durations plus
mailbox/receipt overhead. Do not add host and native durations together.
Manifest `seconds` includes host cleanup and retention; `worker_seconds` is the
native instrumented interval, not complete host wall time.

## Camera and lifecycle

Camera distance follows the selected framing/lens geometry. Yaw rotates the
declared front/right axes; normalized screen placement creates a parallel target
shift, preserving perspective. Lights follow the measured head, not the shifted
camera target. Depth of field is disabled.

Native export may add a static reference-pose root transform. The adapter only
normalizes that selected actor's static root channels to configured placement;
animated or ambiguous roots fail rather than losing motion.

MRQ callbacks retain request-owned objects until completion, then remove only
that job's delegates/resources. The persistent bootstrap keeps the editor alive;
only its owner controls shutdown. Explicit one-shot execution preserves its
separate startup-script lifetime.

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

Technical decoding checks do not certify phoneme alignment, acting or identity.
See [implementation contracts](implementation.md).

## Operational limits

Jobs are synchronous and serialized per alias, not a queue service.
Crashes can leave partial native assets/running manifests; inspect and create a
fresh revision rather than reusing the namespace. Source WAV hashes are checked
again after animation. Existing projects/assets are not silently upgraded.
Managed obsolete packages follow dependency-aware stopped-project retention,
not per-render deletion; external/unregistered assets are protected.

A separate later render currently starts another solve instead of resuming a
saved audio-only Performance. Arm/torso gestures and wardrobe APIs remain outside
the current implementation.

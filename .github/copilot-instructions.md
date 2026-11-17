# Milestone delivery

In every development mode, commit and push each major module milestone once its
relevant checks pass and its remaining limitations are accurately documented.
Use the repository's configured SSH origin and Arka95 identity for pushes.
Keep unrelated work out of milestone commits.

For each completed visual iteration, open its actual new render or an approved-
source/candidate comparison. State what changed and what passed or failed;
distinguish Blender proposals from reloaded Unreal results. Never present an old
render as a new iteration, or a successful tool call as successful likeness.

## Selected Blender starting baseline

On November 17, 2026, the owner selected
`data\arka\facefitting\diagnostics\20261117T0525Z-hairfree-camera\hairfree-matched-camera.blend`
as the working baseline for future accuracy and feature improvements.
Read `data\arka\facefitting\working-baseline.json` and the facefitting guide
before resuming personal-face work. Preserve this scene and its approved source
geometry; better reference samples must lead to new revisions, not overwrites.
Do not refit/regenerate geometry or save additional `.blend` copies unless the
owner explicitly requests them. Refine the existing native Character in Unreal.
Match facial hair to the original photo: dark moustache and short beard/stubble,
denser around the chin/jaw and lighter on the cheeks; do not claim it is applied
until the saved native Character has been visually checked.
Compare improvements with this baseline and show each completed visual iteration.
Do not treat working-baseline selection as final likeness or rigging approval.
The separate Taro reanimation test must not depend on improving this personal head.

## Reanimation iteration quality

The default production portrait preset is 1080x1920, 9:16. Keep resolution
configurable; prefer `--resolution 540x960` for iterative tests unless checking
final-resolution clarity. Resolution changes must not move the approved camera
or silently change lighting, voice or source assets. Label draft previews as
drafts rather than using their softness to judge final quality.
Model Blueprint, narration WAV, resolution, world placement/yaw and normalized
screen anchor must remain independently selectable. Preserve the accepted
default composition; use explicit overrides for placement experiments. Voice
profile/provider selection happens outside ailias. `reanimation run --voice WAV`
and compatibility spelling `--audio WAV` consume supplied audio from any source.
Do not add Yapit, its models or MCP as module/test dependencies or modify Yapit
to implement ailias options. Once these controls work, prioritize personal
MetaHuman generation from the selected Blender working baseline.

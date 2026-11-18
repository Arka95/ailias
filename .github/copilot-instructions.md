# Milestone delivery

In every development mode, commit and push each major module milestone once its
relevant checks pass and its remaining limitations are accurately documented.
Use the repository's configured SSH origin and Arka95 identity for pushes.
Keep unrelated work out of milestone commits.

For each completed visual iteration, open its actual new render or an approved-
source/candidate comparison. State what changed and what passed or failed;
distinguish Blender proposals from reloaded Unreal results. Never present an old
render as a new iteration, or a successful tool call as successful likeness.

## Retired Blender baseline

On November 18, 2026, the owner requested deletion of the selected
`hairfree-matched-camera.blend` baseline and its `working-baseline.json` selector.
They have been removed. Do not restore or use that baseline for new work.
Full pipeline runs reuse retained sampled images and, when requested, original
photo pixels, creating fresh isolated downstream outputs. Preserve original photos and existing native
Characters unless their deletion is explicitly requested.
Match facial hair to the original photo: dark moustache and short beard/stubble,
denser around the chin/jaw and lighter on the cheeks; do not claim it is applied
until the saved native Character has been visually checked.
Compare improvements with original photographs and show each completed visual iteration.
Blender scalp hair is disabled by default; Unreal groom mapping remains independent.
Do not treat a completed pipeline run as final likeness or rigging approval.
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
to implement ailias options. Personal MetaHuman generation must use the current
owner-requested pipeline, not the retired Blender baseline.

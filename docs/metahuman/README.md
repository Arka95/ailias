[Back to REPO_ROOT](../../README.md) | [Development](dev.readme.md)

# MetaHuman assets

Create and edit native Unreal Character assets from verified mesh handoffs or
an existing Character. Geometry conform, appearance, runtime assembly and
narrated animation are distinct operations.

| Operation | Product |
|---|---|
| Static import | Native mesh/material/texture packages; not an editable Character |
| Conform | Editable head-only Character with saved geometry |
| Feature edit | Duplicated Character with explicit skin, iris, groom or bounded sculpt edits |
| Assembly | Compiled runtime Blueprint and compatible rig/assets |
| Neutral review | New native preview without a speech solve |

## Usage

These APIs retain the `facefitting` command namespace:

```powershell
python -m ailias facefitting unreal-import subject APPEARANCE_REVISION --editor $ue --project $project
python -m ailias facefitting unreal-conform subject IMPORT_JOB --editor $ue --project $project
python -m ailias facefitting unreal-features subject --character /Game/Heads/Person.Person --profile C:\profiles\features.json --assemble --editor $ue --project $project
python -m ailias facefitting unreal-review subject FEATURE_JOB --editor $ue --project $project
```

Set `$ue` and `$project` to an installed compatible editor and project.
See the [complete profile/command reference](../facefitting/README.md#configuration).
Use the assembled Blueprint with [reanimation](../reanimation/README.md);
a static mesh or unrigged Character is not a substitute.

## Limits

Conform transfers geometric constraints, not Blender skin, iris or hair.
Stock grooms and native colour pickers are explicit approximations, not measured
photo matching. Save/reload checks verify persistence, not facial resemblance.
Cloud rig/texture requests require the explicit `--cloud-services` flag.

Native operations reuse the project-owned editor where supported. Jobs do not
close it on completion. Obsolete package removal is deferred while Unreal is
open; see [resource ownership](../dev.readme.md#resource-cleanup).

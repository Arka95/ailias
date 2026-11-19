[Back to REPO_ROOT](../../README.md) | [Module guide](README.md)

# MetaHuman internals

## Components and communication

| Component | Responsibility |
|---|---|
| `static_mesh_import.py` | Verified mesh import, package hashes, declared basis/bounds |
| `head_conforming.py` | Optional baseline/photo-camera dependencies and head-only native solve |
| `character_features.py` | Profile validation, immutable-source duplication, edits and optional assembly |
| `appearance_mapping.py` | Translate supported photo evidence/policy into explicit native settings |
| `feature_comparison.py` | Photo/render diagnostics and unapplied iris-profile proposals |
| `character_portrait.py` | Verified runtime review through the reanimation renderer |
| `sculpt_response.py` | Source-pinned finite-difference calibration and exact state restoration |
| `worker_process.py` | Explicit worker/request/terminal-state execution and saved-result checks |
| `*_worker.py` | Unreal-embedded Python operations, separate from host numerical dependencies |

Host calls are synchronous. Migrated operations use the owned session's
request/receipt mailbox; independent verification launches a fresh process.
See [native process protocol](../generation/dev.readme.md#architecture-and-contracts).
Worker sources are copied/hash-pinned into jobs.

## Geometry and persistence

The native target/camera uses `(GNM X, Z, Y) * 100`; static preview import uses
`(GNM Z, -X, Y) * 100`. Neither establishes measured physical scale.
Photo-contour refinement retains the original full-frame camera coordinates.
Never replace them with a centered crop without changing intrinsics.

Edits duplicate the source Character into a new namespace. Geometry changes
invalidate an inherited rig on the copy, not the original. Commit native state,
save packages and verify files independently. Preserve original package hashes
and optional baseline dependencies. Appearance-only edits must preserve geometry.

## Appearance and assembly

The editing preview owns a collection distinct from the Character's internal
collection. Edit `get_preview_collection`, call `on_edit_preview_collection`,
then save; wardrobe slot replacement uses `set_single_slot_selection`.
Reload checks verify requested properties and Skin preview mode.

Skin picker coordinates are not RGB; skin roughness is a multiplier, distinct
from normalized groom roughness. Native face textures require native UVs.
An explicit null base-colour override restores the native complexion.
Unspecified profile fields remain unchanged.

Save validated edits before assembly. Reuse compatible rig/texture data when
available; missing cloud prerequisites fail unless explicitly enabled.
Independent runtime inspection instantiates the compiled actor, checks its
Face/head/RigLogic and requested groom bindings, then destroys only that test
actor. Baked-only short grooms do not satisfy the bound-component gate.

D3D12/SM6 is explicit; NullRHI is not a compatible replacement for rigged
Character editing. Real material compilation failures invalidate publication
even if asset export reports success.

## Resource ownership

Workers release only request-owned actors, callbacks and temporary resources.
They do not stop the persistent editor. Native retention inspects registry
dependencies without loading old assets, and deletes eligible exact files only
after a fully revalidated editor shutdown.
See [retention contract](../dev.readme.md#resource-cleanup).

## Tests

```powershell
python -m unittest discover -s src\ailias\metahuman -t src -q
python -m unittest ailias.generation.tests.test_unreal_session ailias.generation.tests.test_unreal_retention -q
```

Units use fake Unreal objects and temporary packages. They check profile
bounds, source preservation, persistence/assembly failures, session ownership
and retention safety. Actual asset reload/render checks are explicit native
integrations, not unit-test side effects or likeness certification.

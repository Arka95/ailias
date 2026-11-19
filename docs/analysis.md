[Back to REPO_ROOT](../README.md) | [Development](dev.readme.md)

# Design decisions

These decisions describe implementation boundaries. Execution histories and
individual measurements are kept outside the manuals.

| Decision | Chosen option and configuration | Tradeoff / alternative | Acceptance criterion |
|---|---|---|---|
| Interface | Local Python CLI with JSON output | Simple automation; no capture UI or remote service | Stable argument, output and exit-code contracts |
| Persistence | Atomic JSON, hashed artifacts and per-alias writer exclusion | Inspectable local files; no multi-machine coordination | Reject changed dependencies and preserve prior committed state |
| Recognition vs geometry | Separate grouping, image quality and dense geometry evidence | Identity similarity does not establish reconstructability | Preserve uncertain cases for review |
| Photo preparation | Native image grid plus semantic head masks and reversible analysis alignment | Better isolation without synthesized frontal anatomy | Source-coordinate reconstruction and unchanged originals |
| Colour | Bounded scene-neutral correction and explicit observed coverage | Not intrinsic albedo; ambiguous evidence abstains | Do not label filled/clipped texels as measured colour |
| Head fitting | Shared GNM identity, per-view camera/expression; explicit CPU/CUDA policy | Prior-supported hidden anatomy, not a calibrated scan | Finite outputs, consistent transforms and conditional held-out evaluation |
| Search | Optional configured candidates; compatible reference reuse within one search | Higher capacities cost work and can overfit | Preserve candidates, images, failed trials and final all-view fit |
| Photogrammetry | Separate COLMAP route with strict standard gates | Can block or produce fragments instead of a complete head | Publish measured matching/camera/surface evidence, not inferred anatomy |
| Domain structure | `head_reconstruction`, `metahuman`, `pipelines` | More explicit boundaries; public CLI remains compatible | Domain code does not depend on orchestration internals |
| Execution | Named local steps and persisted stage outcomes | No distributed queue or automatic resume cache | Stop on failure; retain dependency and failure evidence |
| Resource ownership | Central finalization, nested retention deferral and reference-aware major products | Conservative retention can defer reclamation | Preserve active, referenced, edited, linked and unknown resources |
| Native lifetime | Project-owned offscreen editor, JSON mailbox and OS writer lock | Resident memory and state require lifecycle discipline | Validate PID identity, one admitted job and non-reentrant dispatch |
| Native verification | Independent saved-asset reload and actual file/media checks | Additional work beyond API return values | Do not publish missing, incompatible or fallback-material assets |
| Native cleanup | Registry inspection followed by stopped-project exact-file deletion | Cannot reclaim eligible package files while editor remains open | Revalidate complete package snapshot and dependency closure |
| Appearance | Explicit native settings or native-UV photo projection | Stock grooms/pickers are not automatic likeness reconstruction | Check persistence separately from visual acceptance |
| Reanimation | Supplied WAV and existing assembled rig; explicit encoder | No speech synthesis or implicit rig creation | Verify frames, source identity and complete media decoding |

The local filesystem is a single-host persistence boundary, not a replicated
database. Distributed CAP classifications do not describe its guarantees.
Concurrency and recovery follow alias/project locks, atomic writes and
revalidated artifact identities.

## Engineering principles

Keep transformations separate from I/O and native runtime effects. Use
descriptive public names, small cohesive domain modules and explicit resource
ownership. Pure functions do not need lifecycle methods. Shared helpers belong
in `common` only when their semantics are domain-independent; Unreal-embedded
workers retain a separate import/runtime boundary.

## References

| Source | Application |
|---|---|
| [Python naming and interface guidance](https://peps.python.org/pep-0008/) | Public/internal interfaces and readable naming |
| [Python context managers](https://docs.python.org/3/library/contextlib.html) | Scoped resource ownership and deterministic cleanup |
| [Single Responsibility Principle](https://blog.cleancoder.com/uncle-bob/2014/05/08/SingleReponsibilityPrinciple.html) | Group code by reason to change, not by arbitrary file size |
| [Epic Python scripting](https://dev.epicgames.com/documentation/en-us/unreal-engine/scripting-the-unreal-editor-using-python) | Engine-owned asset operations and embedded runtime |
| [MetaHuman Creator scripting](https://dev.epicgames.com/documentation/metahuman/metahuman-creator-python-scripting-in-unreal-engine?lang=en-US) | Native Character editing |
| [Native session contracts](generation/dev.readme.md) | Local implementation and failure semantics |
| [Quality acceptance](todo/README.md) | Distinguish technical validity from visual quality |

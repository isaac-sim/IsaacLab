# PhysX SimulationManager Lifecycle Hotfix Design

## Context

PR #6505 removed Isaac Lab's Kit extension manifests and the corresponding
experience-file dependencies. That packaging change means `isaaclab_physx` can
be imported while resolving task configuration, before Kit loads
`isaacsim.core.simulation_manager`.

The existing `_patch_isaacsim_simulation_manager()` compatibility hook is lazy:
it returns when Isaac Sim's module is not yet present. Before #6505, Kit loaded
the `isaaclab_physx` extension again during startup, which gave the hook a
second opportunity to disable Isaac Sim's default lifecycle callbacks. After
#6505, that second import no longer occurs.

Isaac Sim's original STOP callback consequently remains subscribed. It calls
`SimulationManager.invalidate_physics()`, invalidating the tensor view shared
with Isaac Lab's `PhysxManager`. Franka then fails during its first reset with
invalid `updateArticulationsKinematic` and `getDofVelocities` calls.

## Goals

- Make `PhysxManager` claim sole ownership of the PhysX simulation lifecycle
  after Kit has initialized, independent of Python package import order.
- Cover both `launch_simulation()` and direct `AppLauncher` users.
- Preserve pre-Kit config imports without eagerly importing Isaac Sim modules.
- Add a regression test that fails under the #6505 import order.
- Keep the change private and backward compatible.

## Non-goals

- Restore the removed Kit extension manifests.
- Change articulation ordering or tensor-view semantics.
- Introduce a new public lifecycle API.

## Considered approaches

1. **Claim lifecycle ownership in `PhysxManager.initialize()` (selected).**
   Initialization always runs after Kit exists and only when PhysX is the
   selected backend. It also covers callers that construct `AppLauncher` and
   `SimulationContext` directly.
2. **Re-run the hook from `launch_simulation()`.** This is earlier in the
   managed launcher path, but direct `AppLauncher` users bypass it.
3. **Restore extension startup entries.** This recreates the old timing but
   reverses #6505's packaging direction and couples correctness to import side
   effects.

## Design

`PhysxManager.initialize()` will call the existing idempotent compatibility
hook before initializing backend state. At that point
`isaacsim.core.simulation_manager` is loaded, so the hook disables the original
class's PLAY, STOP, stage-open, and stage-close callbacks, then replaces the
module's `SimulationManager` and `IsaacEvents` aliases with Isaac Lab's types.

The hook remains a no-op during pre-Kit package imports. Its documentation will
describe the two-phase behavior: opportunistic patching at package import and
the guaranteed retry during PhysX manager initialization.

## Testing

An integration regression test will import `isaaclab_physx` before launching
Kit, retain the original Isaac Sim manager class after launch, create a
`SimulationContext` with `PhysxCfg`, and assert that initialization:

- redirects the Isaac Sim module alias to `PhysxManager`; and

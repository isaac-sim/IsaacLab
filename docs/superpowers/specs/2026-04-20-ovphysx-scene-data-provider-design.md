# OVPhysX SceneDataProvider — Design Spec

**Issue:** Needs creation — \[OVPHYSX\] SceneDataProvider
**Date:** 2026-04-20
**Status:** Draft

## Summary

Implement `OvPhysxSceneDataProvider` for the OVPhysX backend, satisfying the `BaseSceneDataProvider` contract. This enables Newton-based visualizers (Rerun, Viser) to display OVPhysX simulations by providing body transforms and metadata.

The OVPhysX approach is closer to Newton's implementation (delegation to physics manager) than PhysX's (which builds its own Newton model from USD). OVPhysX has direct access to body transforms via TensorBindings and does not need Kit/Fabric.

## Contract to Satisfy

### BaseSceneDataProvider (source/isaaclab/isaaclab/physics/base_scene_data_provider.py)

| Method | Signature | Purpose |
|--------|-----------|---------|
| `update` | `(env_ids: list[int] \| None) -> None` | Refresh cached transforms |
| `get_newton_model` | `() -> Any \| None` | Newton model handle (for viz) |
| `get_newton_state` | `(env_ids) -> Any \| None` | Newton state handle |
| `get_usd_stage` | `() -> Any \| None` | USD stage handle |
| `get_metadata` | `() -> dict[str, Any]` | Backend metadata |
| `get_transforms` | `() -> dict[str, Any] \| None` | Body positions + orientations |
| `get_velocities` | `() -> dict[str, Any] \| None` | Body velocities |
| `get_contacts` | `() -> dict[str, Any] \| None` | Contact data (can return None) |
| `get_camera_transforms` | `() -> dict[str, Any] \| None` | Camera transforms (can return None) |

## Architecture

### File Layout

```
source/isaaclab_ovphysx/isaaclab_ovphysx/scene_data_providers/
├── __init__.py
├── __init__.pyi
└── ovphysx_scene_data_provider.py  (~400 lines)
```

### Implementation Approach

OVPhysX's situation is unique:
- **Like Newton:** The physics engine is the source of truth for transforms (no Kit/Fabric layer)
- **Unlike Newton:** OVPhysX doesn't have a native Newton model — it uses PhysX underneath
- **Like PhysX:** It may need to build a Newton model from USD for visualization

**Recommended approach:** Follow PhysX's pattern — build a Newton model from the USD stage at initialization, then sync OVPhysX body transforms into Newton state on each `update()` call. This enables all Newton-based visualizers without requiring changes to the visualizer stack.

**ovphysx_scene_data_provider.py:**

1. `__init__(stage, sim_context)`:
   - Store references to USD stage and OvPhysxManager
   - Lazy-build Newton model on first access

2. `update(env_ids)`:
   - Read body transforms from ovphysx TensorBindings
   - Write transforms into Newton model state
   - For partial updates, only sync requested env_ids

3. `get_newton_model()`:
   - Build Newton model from USD if not yet built (same as PhysX approach)
   - Return cached model

4. `get_newton_state(env_ids)`:
   - Return Newton state (synced in `update()`)
   - Filter for env_ids if specified

5. `get_transforms()`:
   - Read body poses from ovphysx bindings
   - Return as `{"positions": ..., "orientations": ...}` dict

6. `get_velocities()`:
   - Read body velocities from ovphysx bindings
   - Return as `{"linear": ..., "angular": ...}` dict

7. `get_usd_stage()`:
   - Return the USD stage handle (OVPhysX loads from USD, so stage is available)

8. `get_metadata()`:
   - Return `{"physics_backend": "ovphysx", "num_envs": N}`

9. `get_contacts()` / `get_camera_transforms()`:
   - Return None (not supported initially)

### OVPhysX API Requirements

| API | Purpose | Notes |
|-----|---------|-------|
| Body transforms | All body poses for visualization | Available via existing bindings |
| Body velocities | Optional velocity visualization | Available via existing bindings |

**No new ovphysx APIs needed** for the core implementation.

### Newton Model Building

The Newton model building logic can be shared with PhysX's implementation:
- Uses `ModelBuilder` to construct from USD
- Handles prebuilt artifacts for fast loading
- Maps body indices to environment indices

Consider extracting the Newton model building code from `PhysxSceneDataProvider` into a shared utility if significant code overlap exists.

## Factory Registration

Add `"ovphysx"` to the factory dispatch in `source/isaaclab/isaaclab/physics/scene_data_provider.py`:

```python
if backend_name == "ovphysx":
    from isaaclab_ovphysx.scene_data_providers import OvPhysxSceneDataProvider
    return OvPhysxSceneDataProvider(stage, sim_context)
```

Also update `source/isaaclab/isaaclab/utils/backend_utils.py` if needed.

## Tests

No existing SceneDataProvider tests to copy. Write basic tests:
- Verify `get_metadata()` returns correct backend name
- Verify `get_transforms()` returns valid body poses after simulation steps
- Verify `update()` syncs transforms correctly

## Dependencies

- **Requires PR #4852 merged**
- **No new ovphysx APIs needed**
- **Depends on Newton model builder** (shared with PhysX)
- **Independent of asset/sensor specs** (reads from physics manager directly)

## Estimated Scope

- `ovphysx_scene_data_provider.py`: ~400 lines
- Factory registration: ~5 lines
- Tests: ~100 lines

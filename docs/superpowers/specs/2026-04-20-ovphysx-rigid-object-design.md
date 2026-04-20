# OVPhysX RigidObject + RigidObjectData — Design Spec

**Issue:** [#5316 — \[OVPHYSX\] RigidObject asset](https://github.com/isaac-sim/IsaacLab/issues/5316)
**Date:** 2026-04-20
**Status:** Draft

## Summary

Implement `RigidObject` and `RigidObjectData` for the OVPhysX backend, satisfying the `BaseRigidObject` and `BaseRigidObjectData` contracts. The implementation mirrors the PhysX RigidObject, substituting PhysX tensor views with ovphysx TensorBindings. The existing OVPhysX Articulation (PR #4852) establishes all patterns to follow.

**Validation environment:** `Isaac-Repose-Cube-Allegro-Direct-v0` — uses an Articulation (Allegro hand, already supported) and a RigidObject (DexCube). Getting this environment running end-to-end validates the RigidObject implementation.

## Guiding Principles

- **Mirror PhysX structure.** File layout, method signatures, and data flow should match the PhysX RigidObject as closely as possible.
- **Reuse Articulation patterns.** The OVPhysX Articulation already implements lazy TensorBinding creation, DLPack zero-copy, CPU-only type routing, timestamped caching, and Warp kernel launches. Reuse these patterns identically.
- **Do not modify contracts.** `BaseRigidObject` and `BaseRigidObjectData` must not be changed. If ovphysx lacks a needed API, flag it as a blocker for @marcodiiga.
- **Tests copied, only setup changes.** PhysX tests are copied with config/backend swaps. Interface tests gain OVPhysX parametrization.

## Contract to Satisfy

### BaseRigidObject (source/isaaclab/isaaclab/assets/rigid_object/base_rigid_object.py)

**Abstract properties (7):**

| Property | Type | Description |
|----------|------|-------------|
| `data` | `RigidObjectData` | Data container |
| `num_instances` | `int` | Number of environments |
| `num_bodies` | `int` | Always 1 |
| `body_names` | `list[str]` | Single body name |
| `root_view` | backend-specific | ovphysx rigid body binding handle |
| `instantaneous_wrench_composer` | `WrenchComposer` | Per-step forces |
| `permanent_wrench_composer` | `WrenchComposer` | Persistent forces |

**Abstract methods — core (4):**

| Method | Signature |
|--------|-----------|
| `reset` | `(env_ids, env_mask) -> None` |
| `write_data_to_sim` | `() -> None` |
| `update` | `(dt: float) -> None` |
| `find_bodies` | `(name_keys, preserve_order) -> (list[int], list[str])` |

**Abstract methods — root pose writers (6):**

| Method | Signature |
|--------|-----------|
| `write_root_pose_to_sim_index` | `(root_pose, env_ids) -> None` |
| `write_root_pose_to_sim_mask` | `(root_pose, env_mask) -> None` |
| `write_root_com_pose_to_sim_index` | `(root_pose, env_ids) -> None` |
| `write_root_com_pose_to_sim_mask` | `(root_pose, env_mask) -> None` |
| `write_root_link_pose_to_sim_index` | `(root_pose, env_ids) -> None` |
| `write_root_link_pose_to_sim_mask` | `(root_pose, env_mask) -> None` |

**Abstract methods — root velocity writers (6):**

| Method | Signature |
|--------|-----------|
| `write_root_velocity_to_sim_index` | `(root_velocity, env_ids) -> None` |
| `write_root_velocity_to_sim_mask` | `(root_velocity, env_mask) -> None` |
| `write_root_com_velocity_to_sim_index` | `(root_velocity, env_ids) -> None` |
| `write_root_com_velocity_to_sim_mask` | `(root_velocity, env_mask) -> None` |
| `write_root_link_velocity_to_sim_index` | `(root_velocity, env_ids) -> None` |
| `write_root_link_velocity_to_sim_mask` | `(root_velocity, env_mask) -> None` |

**Abstract methods — body property setters (6):**

| Method | Signature |
|--------|-----------|
| `set_masses_index` | `(masses, body_ids, env_ids) -> None` |
| `set_masses_mask` | `(masses, body_mask, env_mask) -> None` |
| `set_coms_index` | `(coms, body_ids, env_ids) -> None` |
| `set_coms_mask` | `(coms, body_mask, env_mask) -> None` |
| `set_inertias_index` | `(inertias, body_ids, env_ids) -> None` |
| `set_inertias_mask` | `(inertias, body_mask, env_mask) -> None` |

**Abstract methods — deprecated state writers (3):**

| Method | Signature |
|--------|-----------|
| `write_root_state_to_sim` | `(root_state, env_ids) -> None` |
| `write_root_com_state_to_sim` | `(root_state, env_ids) -> None` |
| `write_root_link_state_to_sim` | `(root_state, env_ids) -> None` |

**Internal hooks (3):**

| Method | Purpose |
|--------|---------|
| `_initialize_impl` | Create ovphysx bindings, data container, buffers |
| `_create_buffers` | Allocate index arrays, wrench composers, defaults |
| `_process_cfg` | Build default root pose/velocity from cfg.init_state |

### BaseRigidObjectData (source/isaaclab/isaaclab/assets/rigid_object/base_rigid_object_data.py)

**Abstract properties (~40):** Root state (pose, velocity in link/com/actor frames), body state (pose, velocity, acceleration), body properties (mass, inertia, com), derived quantities (projected_gravity, heading, body-frame velocities), sliced components (pos, quat, lin_vel, ang_vel extracted from compound types).

**Abstract method (1):** `update(dt: float) -> None`

All properties use warp types: `wp.transformf`, `wp.spatial_vectorf`, `wp.vec3f`, `wp.quatf`, `wp.float32`.

## Architecture

### File Layout

```
source/isaaclab_ovphysx/isaaclab_ovphysx/assets/
├── __init__.py          (add RigidObject, RigidObjectData exports)
├── __init__.pyi         (add type stubs)
├── articulation/        (existing)
└── rigid_object/
    ├── __init__.py
    ├── __init__.pyi
    ├── rigid_object.py       (~800 lines)
    └── rigid_object_data.py  (~600 lines)
```

No separate `kernels.py` — reuse kernels from the articulation module or from a shared `assets/kernels.py` (same as PhysX pattern).

### Implementation Pattern

Follow the existing Articulation exactly:

**rigid_object.py:**
1. `_initialize_impl()`:
   - Get OvPhysxManager instance
   - Resolve rigid body prim path (single body, no articulation root)
   - Create `RigidObjectData` with lazy binding getter (`_get_binding`)
   - Call `_create_buffers()`, `_process_cfg()`, `update(0.0)`
   - Set `data.is_primed = True`

2. `_get_binding(tensor_type)`:
   - Lazy-create TensorBinding from ovphysx instance for the requested `TensorType`
   - Cache in `_bindings` dict
   - Route CPU-only types (mass, com, inertia) appropriately

3. Root state writers (index variants):
   - Convert input to GPU float32 via `_to_flat_f32()`
   - For `write_root_pose_to_sim_index`: write via `RIGID_BODY_ROOT_POSE` binding
   - For `write_root_link_pose_to_sim_index`: compose with COM offset, then write
   - For `write_root_com_pose_to_sim_index`: decompose to link frame, then write
   - Use Warp kernels for frame conversions (same kernels as Articulation)

4. Root state writers (mask variants):
   - Same as index but use mask-based write path

5. `write_data_to_sim()`:
   - Apply external wrenches from both composers via ovphysx force API

6. `reset()`:
   - Write default pose and velocity for specified env_ids
   - Clear wrench composers
   - Reset data timestamps

**rigid_object_data.py:**
1. Same lazy TensorBinding + timestamped buffer caching as ArticulationData
2. Root state properties read from `RIGID_BODY_ROOT_POSE` and `RIGID_BODY_ROOT_VELOCITY` bindings
3. Body state properties are trivial wrappers (num_bodies=1, so body[0] == root)
4. Derived properties use Warp kernels:
   - `projected_gravity_b`: `quat_apply_inverse(quat, gravity_vec)`
   - `heading_w`: `atan2(forward_x, forward_y)` from root orientation
   - `root_link_lin_vel_b`: `quat_apply_inverse(quat, lin_vel_w)`
   - `root_link_ang_vel_b`: `quat_apply_inverse(quat, ang_vel_w)`
5. COM properties compose root link pose with body COM offset

### OVPhysX API Requirements

The implementation needs these TensorType bindings from ovphysx:

| TensorType | Shape | Device | Purpose |
|------------|-------|--------|---------|
| `RIGID_BODY_ROOT_POSE` | `(N, 7)` | GPU | Read/write root transforms |
| `RIGID_BODY_ROOT_VELOCITY` | `(N, 6)` | GPU | Read/write root velocities |
| `RIGID_BODY_MASS` | `(N, 1)` | CPU | Read/write masses |
| `RIGID_BODY_COM_POSE` | `(N, 7)` | CPU | Read/write center-of-mass poses |
| `RIGID_BODY_INERTIA` | `(N, 9)` | CPU | Read/write inertia tensors |

Plus a force/torque application API:
- `add_forces_and_torques_index(forces, torques, indices)` — or the ovphysx equivalent

**Blocker for @marcodiiga:** If `RIGID_BODY_*` TensorType variants don't exist yet in the ovphysx wheel, they need to be added. The Articulation uses `ARTICULATION_*` types. Check whether ovphysx already exposes rigid body bindings or whether this is new work.

### Warp Kernels

Reuse existing kernels from the Articulation module where possible:

| Kernel | Source | Purpose |
|--------|--------|---------|
| `set_root_link_pose_to_sim` | `assets/kernels.py` | Convert link pose for sim write |
| `set_root_com_pose_to_sim` | `assets/kernels.py` | Convert COM pose for sim write |
| `set_root_com_velocity_to_sim` | `assets/kernels.py` | Convert COM velocity for sim write |
| `get_root_link_vel_from_root_com_vel` | `assets/kernels.py` | Frame velocity conversion |
| `get_root_com_pose_from_root_link_pose` | `assets/kernels.py` | Frame pose conversion |
| `quat_apply_inverse_1D_kernel` | `assets/kernels.py` | World-to-body frame rotation |
| `root_heading_w` | `assets/kernels.py` | Heading computation |

If these kernels already exist in the Articulation's `kernels.py`, extract them to a shared `assets/kernels.py` (matching the PhysX pattern where both Articulation and RigidObject use the same `assets/kernels.py`).

### Mock Infrastructure

Extend `MockOvPhysxBindingSet` in `isaaclab_ovphysx/test/mock_interfaces/views/mock_ovphysx_bindings.py`:
- Add rigid body tensor types to the mock
- Create `MockRigidBodyBindingSet` factory method that produces bindings with `num_bodies=1`, no joints

## Tests

### Backend-specific tests (copy from PhysX)

**Source:** `source/isaaclab_physx/test/assets/test_rigid_object.py`
**Target:** `source/isaaclab_ovphysx/test/assets/test_rigid_object.py`

Changes from PhysX version:
- Replace `PhysxCfg` with `OvPhysxCfg` in simulation config
- Replace PhysX-specific USD asset paths if they require Kit/Nucleus (use local test assets)
- Skip tests that require Kit-specific features (AppLauncher, Nucleus)

### Interface tests (add OVPhysX parametrization)

**File:** `source/isaaclab/test/assets/test_rigid_object_iface.py`

Add OVPhysX as a parametrized backend alongside PhysX and Newton:
- Add `"ovphysx"` to the backend parametrize list
- Create mock rigid body bindings for the OVPhysX backend
- Follow the same pattern as `test_articulation_iface.py` which already has OVPhysX

## Validation

### Target environment

**`Isaac-Repose-Cube-Allegro-Direct-v0`**

This environment uses:
- Allegro hand (ArticulationCfg) — already supported by OVPhysX
- DexCube (RigidObjectCfg) — the component being implemented

**Validation steps:**
1. Run the environment with OVPhysX backend for 100 steps
2. Verify RigidObject state reads return plausible values (cube falls under gravity)
3. Verify resets write the cube back to initial pose
4. Verify wrench application works (cube responds to external forces)
5. Run a short training (1000 steps) to verify gradients flow through the environment

### Test commands

```bash
# Interface tests
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_rigid_object_iface.py -k ovphysx

# Backend-specific tests
./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/assets/test_rigid_object.py

# Validation environment
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/allegro_hand_env.py --num_envs 4
```

## Dependencies

- **Requires PR #4852 merged** (OVPhysX Articulation + physics manager)
- **May require ovphysx wheel update** if rigid body TensorTypes are missing
- **No changes to core isaaclab** beyond adding OVPhysX to interface test parametrization

## Estimated Scope

- `rigid_object.py`: ~800 lines (PhysX is ~1000, OVPhysX Articulation patterns reduce boilerplate)
- `rigid_object_data.py`: ~600 lines (PhysX is ~800)
- `kernels.py` changes: extract shared kernels if not already shared
- Mock extensions: ~50 lines
- Test copies + adaptations: ~200 lines of config changes
- Interface test changes: ~20 lines (add parametrization)

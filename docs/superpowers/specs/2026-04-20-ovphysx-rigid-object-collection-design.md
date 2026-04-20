# OVPhysX RigidObjectCollection — Design Spec

**Issue:** [#5317 — \[OVPHYSX\] RigidObjectCollection asset](https://github.com/isaac-sim/IsaacLab/issues/5317)
**Date:** 2026-04-20
**Status:** Draft

## Summary

Implement `RigidObjectCollection` and `RigidObjectCollectionData` for the OVPhysX backend, satisfying the `BaseRigidObjectCollection` and `BaseRigidObjectCollectionData` contracts. This extends the single-body RigidObject to manage multiple distinct rigid bodies per environment, adding dual (env, body) indexing.

**Validation environment:** Franka cube stacking task (`stack_joint_pos_instance_randomize_env_cfg.py`) — uses a Franka arm (Articulation) and multiple cubes (RigidObjectCollection).

**Depends on:** #5316 (RigidObject) — shares kernels and patterns.

## Guiding Principles

Same as RigidObject spec:
- Mirror PhysX structure
- Reuse patterns from OVPhysX Articulation and RigidObject
- Do not modify contracts
- Tests copied with only setup changes

## Key Difference from RigidObject

RigidObjectCollection manages N distinct rigid bodies across M environments. This adds:

1. **Dual indexing** — methods accept both `env_ids` and `body_ids` (or `env_mask` and `body_mask`)
2. **View-order reshaping** — ovphysx tensor bindings likely use body-major ordering `[body0_env0, body0_env1, ..., body1_env0, ...]`. Methods must reshape between instance-order `(num_instances, num_bodies, ...)` and view-order `(num_bodies * num_instances, ...)`
3. **Per-body properties** — mass, inertia, COM are independent per body per environment
4. **Data dimensionality** — all state tensors are `(num_instances, num_bodies, ...)` instead of `(num_instances, ...)`

## Contract to Satisfy

### BaseRigidObjectCollection

**Abstract properties (7):** Same as RigidObject — `data`, `num_instances`, `num_bodies`, `body_names`, `root_view`, `instantaneous_wrench_composer`, `permanent_wrench_composer`.

**Abstract methods — core (4):**
- `reset(env_ids, object_ids, env_mask)` — note extra `object_ids` parameter vs RigidObject
- `write_data_to_sim()`
- `update(dt)`
- `find_bodies(name_keys, preserve_order) -> (torch.Tensor, list[str])` — returns Tensor not list[int]

**Abstract methods — body pose writers (6):**
- `write_body_pose_to_sim_index/mask(body_poses, body_ids, env_ids)`
- `write_body_link_pose_to_sim_index/mask(body_poses, body_ids, env_ids)`
- `write_body_com_pose_to_sim_index/mask(body_poses, body_ids, env_ids)`

**Abstract methods — body velocity writers (6):**
- `write_body_velocity_to_sim_index/mask(body_velocities, body_ids, env_ids)`
- `write_body_link_velocity_to_sim_index/mask(body_velocities, body_ids, env_ids)`
- `write_body_com_velocity_to_sim_index/mask(body_velocities, body_ids, env_ids)`

**Abstract methods — body property setters (6):**
- `set_masses_index/mask(masses, body_ids, env_ids)`
- `set_coms_index/mask(coms, body_ids, env_ids)`
- `set_inertias_index/mask(inertias, body_ids, env_ids)`

**Abstract methods — deprecated (3):**
- `write_body_state_to_sim`, `write_body_com_state_to_sim`, `write_body_link_state_to_sim`

**Internal hooks (4):**
- `_initialize_impl`, `_create_buffers`, `_process_cfg`, `_invalidate_initialize_callback`

### BaseRigidObjectCollectionData

**Abstract properties (~30):** Body state (link/com pose, velocity, acceleration), body properties (mass, inertia, com), derived quantities (projected_gravity, heading, body-frame velocities), sliced components. All shapes are `(num_instances, num_bodies, ...)`.

**Abstract method (1):** `update(dt: float)`

## Architecture

### File Layout

```
source/isaaclab_ovphysx/isaaclab_ovphysx/assets/
├── __init__.py          (add RigidObjectCollection exports)
├── __init__.pyi
├── articulation/        (existing)
├── rigid_object/        (from #5316)
└── rigid_object_collection/
    ├── __init__.py
    ├── __init__.pyi
    ├── rigid_object_collection.py       (~1200 lines)
    └── rigid_object_collection_data.py  (~800 lines)
```

### Implementation Pattern

**rigid_object_collection.py:**

1. `_initialize_impl()`:
   - Get OvPhysxManager instance
   - Resolve all rigid body prim paths (multiple bodies)
   - Create TensorBindings for all bodies as a single batched view
   - Build `_env_body_ids_to_view_ids()` mapping for dual indexing
   - Create `RigidObjectCollectionData` with lazy binding getter

2. View-order reshaping (critical complexity):
   - `reshape_view_to_data(view_tensor)` — convert `(num_bodies * num_instances, D)` to `(num_instances, num_bodies, D)`
   - `reshape_data_to_view(data_tensor)` — reverse
   - `_env_body_ids_to_view_ids(env_ids, body_ids)` — map `(env_id, body_id)` pairs to flat view indices
   - These reshaping functions use Warp kernels for GPU-native operation

3. Body state writers — same pattern as RigidObject but with dual indexing:
   - Accept `(num_selected_envs, num_selected_bodies, D)` input
   - Map to view-order flat indices
   - Write via TensorBinding with flat indices

4. Wrench application — wrenches applied per `(env, body)` pair, reshaped to view order

**rigid_object_collection_data.py:**
- Same lazy binding + timestamped caching pattern
- All properties return `(num_instances, num_bodies, ...)` tensors
- Derived properties (gravity projection, heading, body-frame velocities) computed per body

### OVPhysX API Requirements

Same tensor types as RigidObject, but the bindings must handle multiple bodies:
- The ovphysx binding shape becomes `(num_bodies * num_instances, D)` in view order
- Need to confirm ovphysx supports batched rigid body views across multiple distinct prims

**Blocker for @marcodiiga:** Confirm that ovphysx TensorBindings can create a single view spanning multiple distinct rigid body prims (not just clones of the same prim). This is how PhysX's `RigidBodyView` works with a glob expression matching multiple body paths.

### Warp Kernels

In addition to kernels shared with RigidObject:

| Kernel | Purpose |
|--------|---------|
| `reshape_view_to_data_2d` | Reorder body-major to instance-major |
| `reshape_data_to_view_2d` | Reorder instance-major to body-major |
| `env_body_to_view_scatter` | Scatter write for dual-indexed updates |

## Tests

### Backend-specific tests (copy from PhysX)

**Source:** `source/isaaclab_physx/test/assets/test_rigid_object_collection.py`
**Target:** `source/isaaclab_ovphysx/test/assets/test_rigid_object_collection.py`

Changes: swap `PhysxCfg` for `OvPhysxCfg`, adapt USD asset paths.

### Interface tests

**File:** `source/isaaclab/test/assets/test_rigid_object_collection_iface.py`

Add `"ovphysx"` backend parametrization with mock bindings.

## Validation

**Franka cube stacking** — exercises RigidObjectCollection with multiple cubes, Franka arm (Articulation), and FrameTransformer sensor.

**Validation steps:**
1. Run stacking env with OVPhysX for 100 steps
2. Verify each cube in the collection has independent state
3. Verify dual-indexed resets (reset specific cubes in specific envs)
4. Verify body property reads return correct per-body values

## Dependencies

- **Requires #5316 (RigidObject)** — shares kernels and patterns
- **Requires PR #4852 merged**
- **May require ovphysx wheel update** for multi-body rigid body views

## Estimated Scope

- `rigid_object_collection.py`: ~1200 lines
- `rigid_object_collection_data.py`: ~800 lines
- Reshape kernels: ~100 lines
- Tests: ~250 lines of adaptations

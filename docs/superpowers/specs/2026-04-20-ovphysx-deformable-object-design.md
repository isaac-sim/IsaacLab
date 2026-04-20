# OVPhysX DeformableObject — Design Spec

**Issue:** Needs creation — \[OVPHYSX\] DeformableObject asset
**Date:** 2026-04-20
**Status:** Draft

## Summary

Implement `DeformableObject` and `DeformableObjectData` for the OVPhysX backend, enabling soft body simulation.

**Critical finding:** Unlike RigidObject and Articulation, `DeformableObject` has **no base abstract class** in core `isaaclab/`. It exists only in `isaaclab_physx/` and inherits directly from `AssetBase`. This means the standard factory pattern cannot be used. A `BaseDeformableObject` must be extracted first.

## Prerequisite: Extract BaseDeformableObject

Before OVPhysX can implement DeformableObject, the existing PhysX implementation must be refactored:

1. **Create `BaseDeformableObject`** in `source/isaaclab/isaaclab/assets/deformable_object/`:
   - Extract the abstract interface from PhysX's `DeformableObject`
   - Define abstract properties: `data`, `num_instances`, `num_bodies`, `root_view`, `max_sim_vertices_per_body`, `max_sim_elements_per_body`, `max_collision_vertices_per_body`, `max_collision_elements_per_body`
   - Define abstract methods: `write_nodal_state_to_sim_index`, `write_nodal_pos_to_sim_index`, `write_nodal_velocity_to_sim_index`, `write_nodal_kinematic_target_to_sim_index`, `transform_nodal_pos`
   - Inherit from `AssetBase`

2. **Create `BaseDeformableObjectData`** in the same location:
   - Abstract properties: `nodal_pos_w`, `nodal_vel_w`, `nodal_state_w`, `root_pos_w`, `root_vel_w`
   - Attributes: `default_nodal_state_w`, `nodal_kinematic_target`

3. **Refactor PhysX's `DeformableObject`** to inherit from `BaseDeformableObject` instead of `AssetBase`

4. **Wire up factory pattern** in `backend_utils.py` for deformable objects

This is a **breaking change in architecture** (not API) and should be reviewed independently before the OVPhysX implementation begins.

## DeformableObject Interface

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `data` | `DeformableObjectData` | Nodal state container |
| `num_instances` | `int` | Number of deformable instances |
| `num_bodies` | `int` | Always 1 |
| `root_view` | backend-specific | Deformable body view handle |
| `max_sim_vertices_per_body` | `int` | Max simulation vertices |
| `max_sim_elements_per_body` | `int` | Max simulation elements |
| `max_collision_vertices_per_body` | `int` | Max collision vertices |
| `max_collision_elements_per_body` | `int` | Max collision elements |

### Key Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `write_nodal_state_to_sim_index` | `(nodal_state, env_ids, full_data) -> None` | Write nodal positions + velocities |
| `write_nodal_pos_to_sim_index` | `(nodal_pos, env_ids, full_data) -> None` | Write positions only |
| `write_nodal_velocity_to_sim_index` | `(nodal_vel, env_ids, full_data) -> None` | Write velocities only |
| `write_nodal_kinematic_target_to_sim_index` | `(targets, env_ids, full_data) -> None` | Set kinematic targets (volume deformables only) |
| `transform_nodal_pos` | `(nodal_pos, pos, quat) -> Tensor` | Apply rigid transform to nodes |

### Data Properties

| Property | Shape | Type | Description |
|----------|-------|------|-------------|
| `nodal_pos_w` | `(N, V)` | `wp.vec3f` | Per-vertex world positions [m] |
| `nodal_vel_w` | `(N, V)` | `wp.vec3f` | Per-vertex world velocities [m/s] |
| `nodal_state_w` | `(N, V)` | `wp.vec6f` | Combined position + velocity |
| `root_pos_w` | `(N, 3)` | `wp.float32` | Mean nodal position (center) |
| `root_vel_w` | `(N, 3)` | `wp.float32` | Mean nodal velocity |

Where N=num_instances, V=max_sim_vertices_per_body.

## Architecture

### File Layout

```
source/isaaclab_ovphysx/isaaclab_ovphysx/assets/
└── deformable_object/
    ├── __init__.py
    ├── __init__.pyi
    ├── deformable_object.py       (~500 lines)
    ├── deformable_object_data.py  (~300 lines)
    └── kernels.py                 (~100 lines)
```

### OVPhysX API Requirements

| API | Purpose | Notes |
|-----|---------|-------|
| Deformable body view | Access nodal state | Likely new in ovphysx |
| Nodal position read/write | State access | Per-vertex GPU tensors |
| Nodal velocity read/write | Velocity control | Per-vertex GPU tensors |
| Kinematic target API | Partial kinematic control | Volume deformables only |
| Mesh topology query | Vertex/element counts | At initialization |

**Blocker for @marcodiiga:** Deformable body support in ovphysx is likely completely new work. This includes:
- FEM/soft body simulation in the ovphysx backend
- TensorBindings for nodal state (positions, velocities)
- Kinematic target support for volume deformables
- Mesh topology queries

This is the **largest single piece of ovphysx-side work** across all subtasks.

## Phasing

Given the prerequisites, this should be implemented in two phases:

**Phase 1:** Extract `BaseDeformableObject` from PhysX (prerequisite PR)
**Phase 2:** Implement OVPhysX DeformableObject (depends on ovphysx soft body support)

## Tests

**Source:** `source/isaaclab_physx/test/assets/test_deformable_object.py`
**Target:** `source/isaaclab_ovphysx/test/assets/test_deformable_object.py`

## Dependencies

- **Requires base class extraction** (new PR, must merge before implementation)
- **Requires ovphysx soft body support** (significant ovphysx-side work)
- **Requires PR #4852 merged**
- **Independent of other OVPhysX asset specs**

## Estimated Scope

- Base class extraction: ~500 lines of refactoring (separate PR)
- `deformable_object.py`: ~500 lines
- `deformable_object_data.py`: ~300 lines
- `kernels.py`: ~100 lines
- Tests: ~200 lines of adaptations
- **ovphysx-side work**: very significant — FEM support is new

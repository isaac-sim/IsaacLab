# OVPhysX ContactSensor — Design Spec

**Issue:** Needs creation — \[OVPHYSX\] ContactSensor
**Date:** 2026-04-20
**Status:** Draft

## Summary

Implement `ContactSensor` and `ContactSensorData` for the OVPhysX backend, satisfying the `BaseContactSensor` and `BaseContactSensorData` contracts. This is the most complex sensor, requiring contact force reporting, history tracking, air/contact time computation, and optional force matrix filtering.

**Validation environment:** `Isaac-Velocity-Flat-Anymal-C-Direct-v0` — uses an Anymal C robot (Articulation) with ContactSensor on all body links for foot contact detection. Also validates #5321 (rough terrain locomotion) once working on flat terrain.

**Current state:** `isaaclab_ovphysx/test/sensors/check_contact_sensor.py` has 5 stubbed/skipped tests with "Contact sensor not yet supported by ovphysx backend."

## Guiding Principles

- Mirror PhysX `ContactSensor` structure
- Do not modify `BaseContactSensor` or `BaseContactSensorData` contracts
- If ovphysx lacks contact reporting APIs, flag as blocker for @marcodiiga
- Tests copied from PhysX with setup changes

## Contract to Satisfy

### BaseContactSensor (source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor.py)

**Abstract properties (5):**

| Property | Type | Description |
|----------|------|-------------|
| `num_instances` | `int \| None` | Total sensor instances (sensors_per_env * num_envs) |
| `data` | `BaseContactSensorData` | Contact data container |
| `num_sensors` | `int` | Sensor bodies per environment |
| `body_names` | `list[str] \| None` | Ordered sensor body names |
| `contact_view` | backend-specific | Contact view handle |

**Abstract methods (5):**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `reset` | `(env_ids, env_mask) -> None` | Zero buffers for specified envs |
| `compute_first_contact` | `(dt, abs_tol) -> wp.array` | Bodies that made contact within dt |
| `compute_first_air` | `(dt, abs_tol) -> wp.array` | Bodies that broke contact within dt |
| `_create_buffers` | `() -> None` | Allocate all data arrays |
| `_initialize_impl` | `() -> None` | Setup contact views and resolve bodies |
| `_update_buffers_impl` | `(env_mask) -> None` | Fetch and process contact data |

### BaseContactSensorData

**Abstract properties (~14):**

| Property | Shape | Type | Optional |
|----------|-------|------|----------|
| `pose_w` | `(N, S)` | `wp.transformf` | Yes |
| `pos_w` | `(N, S)` | `wp.vec3f` | Yes |
| `quat_w` | `(N, S)` | `wp.quatf` | Yes |
| `net_forces_w` | `(N, S)` | `wp.vec3f` | No |
| `net_forces_w_history` | `(N, H, S)` | `wp.vec3f` | No |
| `force_matrix_w` | `(N, S, F)` | `wp.vec3f` | Yes (if filter_prim_paths_expr) |
| `force_matrix_w_history` | `(N, H, S, F)` | `wp.vec3f` | Yes |
| `contact_pos_w` | `(N, S, F)` | `wp.vec3f` | Yes |
| `friction_forces_w` | `(N, S, F)` | `wp.vec3f` | Yes |
| `last_air_time` | `(N, S)` | `wp.float32` | Yes |
| `current_air_time` | `(N, S)` | `wp.float32` | Yes |
| `last_contact_time` | `(N, S)` | `wp.float32` | Yes |
| `current_contact_time` | `(N, S)` | `wp.float32` | Yes |

Where N=num_envs, S=num_sensors, H=history_length, F=num_filter_shapes.

## Architecture

### File Layout

```
source/isaaclab_ovphysx/isaaclab_ovphysx/sensors/
├── __init__.py
├── __init__.pyi
├── kernels.py                    (shared sensor kernels)
└── contact_sensor/
    ├── __init__.py
    ├── __init__.pyi
    ├── contact_sensor.py         (~600 lines)
    └── contact_sensor_data.py    (~300 lines)
```

### Implementation Pattern

**contact_sensor.py:**

1. `_initialize_impl()`:
   - Get OvPhysxManager instance
   - Find rigid bodies with contact reporting enabled (prim path matching)
   - Create contact data view from ovphysx
   - Resolve sensor count: `num_sensors = total_contacts / num_envs`
   - Resolve filter bodies if `filter_prim_paths_expr` is set
   - Call `_create_buffers()`

2. `_create_buffers()`:
   - Allocate `net_forces_w`: `(num_envs, num_sensors)` dtype `wp.vec3f`
   - Allocate `net_forces_w_history`: `(num_envs, history_length, num_sensors)` dtype `wp.vec3f`
   - Optionally allocate force_matrix, contact_pos, friction_forces, air/contact time arrays
   - Optionally allocate pose tracking arrays

3. `_update_buffers_impl(env_mask)`:
   - Fetch net contact forces from ovphysx contact API
   - Launch `update_net_forces_kernel`: copy forces, shift history, update air/contact timers
   - If pose tracking enabled: fetch body transforms, split to pos/quat
   - If force matrix enabled: fetch filtered contact forces, unpack via kernel
   - If contact points/friction enabled: fetch and unpack

4. `compute_first_contact(dt, abs_tol)`:
   - Launch `compute_first_transition_kernel` to detect contact→no-contact transitions within dt

5. `compute_first_air(dt, abs_tol)`:
   - Same kernel with inverted logic

### OVPhysX API Requirements

This is the **highest-risk component** — contact reporting is not yet implemented in ovphysx.

| API | Purpose | PhysX Equivalent |
|-----|---------|-----------------|
| Contact force query | Net forces per body per step | `RigidContactView.get_net_contact_forces(dt)` |
| Filtered contact forces | Per-pair forces (body vs filter body) | `RigidContactView.get_contact_force_matrix(dt)` |
| Contact point data | Contact positions | `RigidContactView.get_contact_data(dt)` |
| Friction force data | Tangential forces | `RigidContactView.get_friction_data(dt)` |

**Blocker for @marcodiiga:** Contact reporting is the primary missing API in ovphysx. At minimum, `get_net_contact_forces()` is required. Force matrix, contact points, and friction are optional features that can be added incrementally. This is likely the biggest single piece of work on the ovphysx side.

### Warp Kernels

Reuse the same kernel patterns as PhysX (these are physics-engine-agnostic):

| Kernel | Purpose |
|--------|---------|
| `reset_contact_sensor_kernel` | Zero buffers for specified envs |
| `update_net_forces_kernel` | Copy forces, manage history, update timers |
| `compute_first_transition_kernel` | Detect contact/air transitions |
| `unpack_contact_buffer_data` | Aggregate contact points or friction per filter |
| `split_flat_pose_to_pos_quat` | Unpack transform to position + quaternion |

## Tests

### Backend-specific tests (copy from PhysX)

**Source:** `source/isaaclab_physx/test/sensors/test_contact_sensor.py`
**Target:** `source/isaaclab_ovphysx/test/sensors/test_contact_sensor.py`

Also update existing `source/isaaclab_ovphysx/test/sensors/check_contact_sensor.py` — replace skipped stubs with real tests.

### Interface tests

No sensor interface tests exist currently. If one is created, add OVPhysX parametrization.

## Validation

**`Isaac-Velocity-Flat-Anymal-C-Direct-v0`:**
- Anymal C robot with contact sensors on all bodies
- Validate foot contact detection during locomotion
- Verify `compute_first_contact` / `compute_first_air` work for gait detection
- History buffer tracks force evolution correctly

## Dependencies

- **Requires PR #4852 merged**
- **Requires ovphysx contact reporting API** — this is new work for @marcodiiga
- **Depends on #5316 (RigidObject)** for body discovery patterns

## Estimated Scope

- `contact_sensor.py`: ~600 lines
- `contact_sensor_data.py`: ~300 lines
- Kernels: ~200 lines (mostly reusable from PhysX patterns)
- Tests: ~300 lines of adaptations
- **ovphysx-side work**: significant — contact reporting API is new

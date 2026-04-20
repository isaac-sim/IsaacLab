# OVPhysX FrameTransformer Sensor — Design Spec

**Issue:** [#5320 — \[OVPHYSX\] FrameTransformer sensor](https://github.com/isaac-sim/IsaacLab/issues/5320)
**Date:** 2026-04-20
**Status:** Draft

## Summary

Implement `FrameTransformer` and `FrameTransformerData` for the OVPhysX backend, satisfying the `BaseFrameTransformer` and `BaseFrameTransformerData` contracts. Computes relative transforms between a source frame and multiple target frames attached to rigid bodies. Only requires body transforms (no velocities or accelerations).

**Validation environment:** `Isaac-Lift-Cube-Franka-v0` (manager-based) — uses FrameTransformer to track end-effector pose relative to robot base. Also validates #5322 (manipulation support).

## Contract to Satisfy

### BaseFrameTransformer

Inherits `SensorBase`. Additional abstract:

| Property/Method | Type | Description |
|-----------------|------|-------------|
| `data` | `BaseFrameTransformerData` | Transform data |
| `num_bodies` | `int` | Number of target frames (deprecated) |
| `body_names` | `list[str]` | Target frame names (deprecated) |
| `_initialize_impl()` | method | Resolve frames, build index maps |
| `_update_buffers_impl(env_mask)` | method | Fetch transforms, compute relative poses |

### BaseFrameTransformerData

| Property | Shape | Type | Description |
|----------|-------|------|-------------|
| `target_frame_names` | — | `list[str]` | Names of target frames |
| `target_pose_source` | `(N, M)` | `wp.transformf` | Target pose relative to source |
| `target_pos_source` | `(N, M)` | `wp.vec3f` | Target position relative to source |
| `target_quat_source` | `(N, M)` | `wp.quatf` | Target orientation relative to source |
| `target_pose_w` | `(N, M)` | `wp.transformf` | Target pose in world frame (with offset) |
| `target_pos_w` | `(N, M)` | `wp.vec3f` | Target world position |
| `target_quat_w` | `(N, M)` | `wp.quatf` | Target world orientation |
| `source_pose_w` | `(N,)` | `wp.transformf` | Source pose in world frame (with offset) |
| `source_pos_w` | `(N,)` | `wp.vec3f` | Source world position |
| `source_quat_w` | `(N,)` | `wp.quatf` | Source world orientation |

Where N=num_envs, M=num_target_frames.

## Architecture

### File Layout

```
source/isaaclab_ovphysx/isaaclab_ovphysx/sensors/
└── frame_transformer/
    ├── __init__.py
    ├── __init__.pyi
    ├── frame_transformer.py       (~500 lines)
    ├── frame_transformer_data.py  (~150 lines)
    └── kernels.py                 (~100 lines)
```

### Implementation Pattern

**frame_transformer.py:**

1. `_initialize_impl()` (most complex part):
   - Resolve source prim path and verify rigid body exists
   - Resolve all target frame prim paths and verify rigid bodies
   - Compute fixed offsets for source and target frames (once)
   - Build body tracking: create a combined view of all tracked rigid bodies
   - Build per-env index arrays:
     - `_source_raw_indices`: maps env → position in raw transform array
     - `_target_raw_indices`: maps (env, frame) → position in raw transform array
   - Pre-compute offset arrays as GPU warp arrays
   - Allocate output buffers

2. `_update_buffers_impl(env_mask)`:
   - Read all tracked body transforms from ovphysx bindings (single batched read)
   - Launch `frame_transformer_update_kernel` with 2D grid `(num_envs, num_target_frames)`:
     - Fetch source transform by index, apply source offset
     - Fetch target transform by index, apply target offset
     - Compute relative: `target_relative = source_inverse * target_offset`
     - Store 6 outputs (source and target poses in world + relative)

### OVPhysX API Requirements

| API | Purpose | Notes |
|-----|---------|-------|
| Body transforms | Rigid body poses for all tracked frames | Available via existing bindings |

**No new ovphysx APIs needed** — only body transforms are required, which are already available through existing Articulation and (future) RigidObject bindings.

The key implementation question is how to create a single batched view spanning multiple distinct body paths. In PhysX, `RigidBodyView` accepts a glob expression. For ovphysx, the approach depends on whether TensorBindings can be queried by body index within an articulation, or whether a separate mechanism is needed.

**Question for @marcodiiga:** How to read transforms for arbitrary bodies (not just the root) from ovphysx? The Articulation's `ARTICULATION_BODY_POSE` binding returns all link poses — FrameTransformer needs to index into specific links by name.

### Warp Kernels

| Kernel | Purpose |
|--------|---------|
| `frame_transformer_update_kernel` | 2D launch (env, frame): offset application + relative transform computation |

## Tests

**Source:** `source/isaaclab_physx/test/sensors/test_frame_transformer.py`
**Target:** `source/isaaclab_ovphysx/test/sensors/test_frame_transformer.py`

## Validation

**`Isaac-Lift-Cube-Franka-v0`:**
- FrameTransformer tracks Franka end-effector relative to base
- Verify source/target poses match expected robot kinematics
- Verify relative transforms are consistent with forward kinematics

## Dependencies

- **Requires PR #4852 merged**
- **No new ovphysx APIs needed** (uses body transforms from Articulation bindings)
- **Independent of other sensor specs** — can be implemented in parallel

## Estimated Scope

- `frame_transformer.py`: ~500 lines (initialization is complex, update is simple)
- `frame_transformer_data.py`: ~150 lines
- `kernels.py`: ~100 lines
- Tests: ~200 lines of adaptations

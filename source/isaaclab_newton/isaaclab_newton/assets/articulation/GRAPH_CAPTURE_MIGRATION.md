# Articulation Data: CUDA Graph Capture Migration

## Problem

`ArticulationData` uses lazy `TimestampedWarpBuffer` properties that are incompatible with
`wp.ScopedCapture` / CUDA graph capture. When MDP terms access these properties inside a
captured scope, the compute kernel is skipped (timestamp already fresh from warmup) and
never recorded in the graph. Subsequent replays read stale data.

## Property Tiers

### Tier 1 — Sim-bind raw buffers (graph-safe)

Direct physics solver outputs. Updated in-place each `sim.step()`. Stable pointers.

| Property | Type | Source |
|----------|------|--------|
| `_sim_bind_root_link_pose_w` | `wp.transformf` | Solver root pose |
| `_sim_bind_root_com_vel_w` | `wp.spatial_vectorf` | Solver root COM velocity |
| `_sim_bind_body_link_pose_w` | `wp.transformf (2D)` | Solver body poses |
| `_sim_bind_body_com_vel_w` | `wp.spatial_vectorf (2D)` | Solver body COM velocities |
| `_sim_bind_joint_pos` | `wp.float32 (2D)` | Solver joint positions |
| `_sim_bind_joint_vel` | `wp.float32 (2D)` | Solver joint velocities |

### Tier 2 — Derived properties (graph-hostile)

Computed from Tier 1 via `wp.launch`, guarded by `TimestampedWarpBuffer` timestamp check.
The timestamp guard is a Python `if` that prevents the kernel from being captured.

| Property | Computation | Inputs |
|----------|-------------|--------|
| `projected_gravity_b` | Rotate gravity into body frame | `root_link_pose_w` |
| `heading_w` | Extract yaw from quaternion | `root_link_pose_w` |
| `root_link_vel_w` | Project COM vel to link frame | `root_com_vel_w`, `root_link_pose_w` |
| `root_link_vel_b` | Project link vel to body frame | `root_link_vel_w`, `root_link_pose_w` |
| `root_com_vel_b` | Project COM vel to body frame | `root_com_vel_w`, `root_link_pose_w` |
| `root_com_pose_w` | Apply COM offset to link pose | `root_link_pose_w`, `body_com_pos_b` |
| `root_com_acc_w` | Finite difference of COM vel | `root_com_vel_w`, previous vel |
| `body_link_vel_w` | Project body COM vel to link frame | `body_com_vel_w`, `body_link_pose_w` |
| `body_com_pose_w` | Apply COM offset to body poses | `body_link_pose_w`, `body_com_pos_b` |
| `body_com_acc_w` | Finite difference of body vel | `body_com_vel_w`, previous vel |
| `joint_acc` | Finite difference of joint vel | `joint_vel`, previous vel |

### Tier 3 — Sliced properties (mostly graph-safe)

Extract a single component from a compound type. If data is contiguous, a strided
`wp.array` view is created once (zero-cost). If not contiguous, a `wp.launch(split_...)`
runs each access — which IS captured correctly since `is_contiguous` is a fixed flag.

**Exception:** Tier 3 properties that chain through Tier 2 are NOT graph-safe:

| Property | Chains through | Graph-safe? |
|----------|---------------|-------------|
| `root_link_pos_w` | Tier 1 (`_sim_bind_root_link_pose_w`) | Yes |
| `root_link_quat_w` | Tier 1 (`_sim_bind_root_link_pose_w`) | Yes |
| `root_com_lin_vel_w` | Tier 1 (`_sim_bind_root_com_vel_w`) | Yes |
| `root_com_ang_vel_w` | Tier 1 (`_sim_bind_root_com_vel_w`) | Yes |
| `root_link_lin_vel_b` | **Tier 2** (`root_link_vel_b`) | **No** |
| `root_link_ang_vel_b` | **Tier 2** (`root_link_vel_b`) | **No** |
| `root_com_lin_vel_b` | **Tier 2** (`root_com_vel_b`) | **No** |
| `root_com_ang_vel_b` | **Tier 2** (`root_com_vel_b`) | **No** |

## Why Lazy?

The laziness exists for two reasons:

1. **Avoid unnecessary computation.** An env using only `joint_pos` should not pay for
   `projected_gravity_b`. Most envs only use a small subset of derived properties.

2. **Deduplicate within a step.** If multiple MDP terms access `projected_gravity_b` in the
   same step, the timestamp guard ensures the kernel runs only once. Without it, the same
   transform would be recomputed per access.

`update()` (called each `scene.update(dt)`) only eagerly pre-computes `joint_acc` and
`body_com_acc_w` because these need the previous-step velocity snapshot for finite differencing.
Everything else stays lazy.

## Capture Failure Mechanism

```
_wp_capture_or_launch:
  1. WARMUP (eager):
     - MDP term accesses asset.data.projected_gravity_b
     - TimestampedWarpBuffer: timestamp(-1) < sim_timestamp(T) → True
     - wp.launch(project_vec_from_pose_single, ...) runs
     - timestamp set to T

  2. CAPTURE (wp.ScopedCapture):
     - MDP term accesses asset.data.projected_gravity_b
     - TimestampedWarpBuffer: timestamp(T) < sim_timestamp(T) → False
     - wp.launch SKIPPED — kernel NOT recorded in graph
     - MDP term's own wp.launch recorded, pointing to projected_gravity_b.data

  3. REPLAY (all subsequent steps):
     - Only MDP term's kernel replays
     - Reads from projected_gravity_b.data — NEVER recomputed
     - Data is stale from warmup
```

## Key Insight: Tier 2 Kernels ARE Capturable

The preparation kernels (`project_vec_from_pose_single`, `project_velocities_to_frame`,
`compute_heading`) are plain `@wp.kernel` with no Python conditionals. They are fully
capturable. The ONLY problem is the Python `if timestamp < sim_timestamp` guard.

`scene.update()` runs outside any `wp.ScopedCapture` scope. Kernels launched there
execute eagerly every step. MDP terms then read from pre-computed `.data` buffers
(stable pointers), which is capturable.

## Affected MDP Terms

See "Non-Capturable MDP Terms" in `isaaclab_experimental/envs/mdp/WARP_MIGRATION_GAP_ANALYSIS.md`.

Per-step Tier 2 access counts for tested envs:

| Env | `root_com_vel_b` | `projected_gravity_b` | `body_link_vel_w` | Total |
|-----|---:|---:|---:|---:|
| Cartpole | 0 | 0 | 0 | 0 |
| Reach-Franka | 0 | 0 | 0 | 0 |
| Humanoid/Ant | 2 | 2 | 0 | 4 |
| Quadruped velocity | 6 | 2 | 0 | 8 |
| Biped velocity (G1/H1) | 4 | 2 | 1 | 7 |

## Fix Plan

### Phase 1: Inline Tier 1 access in MDP kernels (applied)

Rewrite affected MDP kernels to consume Tier 1 compound types directly
(`root_link_pose_w` as `wp.transformf`, `root_com_vel_w` as `wp.spatial_vectorf`)
and perform the frame rotation inline. Remove `@warp_capturable(False)`.

This is viable because the affected MDP terms do minimal work on top of the
derived property — observations are pure format copies (`vec3f → float32[3]`),
rewards extract a component and do a simple op (square, exp, threshold). Folding
the rotation into the same kernel adds negligible cost and eliminates the Tier 2
dependency entirely.

No changes to `ArticulationData`. All managers become fully capturable.

### Phase 2: Fix lazy update for graph capture (future)

If `ArticulationData` Tier 2 properties are made graph-safe (e.g. via unconditional
materialization in `update()` or selective on-demand computation), MDP terms can
revert to the simpler pattern of reading pre-computed `.data` buffers. This would
be preferable when many MDP terms access the same derived property per step, as it
avoids redundant inline rotations.

The Tier 2 kernels themselves are fully capturable `@wp.kernel` — only the Python
timestamp guard needs removal. A fused kernel in `update()` computing all (or
selectively needed) derived properties in one launch would make Tier 2 graph-safe
with minimal overhead.

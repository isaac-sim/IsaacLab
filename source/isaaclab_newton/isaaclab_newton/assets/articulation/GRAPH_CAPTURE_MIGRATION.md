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

## Proposed Fix: `materialize_derived()`

Add a method to `ArticulationData` that unconditionally launches all Tier 2 kernels
and updates timestamps. Call from `scene.update()` which runs outside capture scopes.

```python
# ArticulationData
def materialize_derived(self) -> None:
    """Eagerly compute all Tier 2 derived properties.

    Call before any captured graph that reads derived data.
    Safe to call every step — cost is the same as accessing each property once.
    """
    # Root-level derived
    _ = self.projected_gravity_b    # forces timestamp check → launches if stale
    _ = self.heading_w
    _ = self.root_link_vel_w
    _ = self.root_link_vel_b
    _ = self.root_com_vel_b
    _ = self.root_com_pose_w
    # Body-level derived
    _ = self.body_link_vel_w
    _ = self.body_com_pose_w
```

Integration point — `scene.update()` or `ArticulationData.update()`:

```python
def update(self, dt: float):
    self._sim_timestamp += dt
    # Existing: finite-difference quantities (need previous-step snapshot)
    self.joint_acc
    self.body_com_acc_w
    # NEW: eagerly materialize all derived properties for graph capture
    self.materialize_derived()
```

**Trade-off:** This removes the lazy optimization — every derived property computes
every step, even if unused. For capture-mode envs this is the correct trade-off (the
kernel cost is negligible vs graph replay savings). For non-capture envs, the extra
kernels add overhead for unused properties.

**Better approach — opt-in materialization:**

Only materialize properties that the env actually uses. The `ManagerCallSwitch` knows
which managers are in capture mode. The env can call `materialize_derived()` only when
capture mode is active:

```python
# In ManagerBasedRLEnvWarp, after scene.update():
if any_manager_in_capture_mode:
    for articulation in self.scene.articulations.values():
        articulation.data.materialize_derived()
```

Or more selectively, track which properties were accessed during warmup and only
materialize those on subsequent steps.

## Alternative: Use Compound Types in MDP Kernels

Instead of fixing the data class, modify MDP terms to use Tier 1 compound types directly
(`root_link_pose_w` as `wp.transformf`, `root_com_vel_w` as `wp.spatial_vectorf`) and
extract components inside warp kernels:

```python
@wp.kernel
def _projected_gravity_kernel(
    pose_w: wp.array(dtype=wp.transformf),
    gravity: wp.vec3f,
    out: wp.array(dtype=wp.float32, ndim=2),
):
    i = wp.tid()
    q = wp.transform_get_rotation(pose_w[i])
    g_b = wp.quat_rotate_inv(q, gravity)
    out[i, 0] = g_b[0]
    out[i, 1] = g_b[1]
    out[i, 2] = g_b[2]
```

**Pros:** No changes to articulation data class. Eliminates all Tier 2/3 overhead.
**Cons:** Every MDP term must be rewritten. Duplicates split logic across terms.

## Affected MDP Terms

See "Non-Capturable MDP Terms" section in
`isaaclab_experimental/envs/mdp/WARP_MIGRATION_GAP_ANALYSIS.md` for the full list of
MDP terms marked `@warp_capturable(False)` due to Tier 2 access, and the pending fix
(`materialize_derived()`) that would make them capturable again.

## Recommendation

Short-term: Mark affected MDP terms `@warp_capturable(False)` so they fall back to
mode=1 automatically. No incorrect results, modest perf regression for those terms.

Medium-term: Add `materialize_derived()` to `ArticulationData` and call it from
`scene.update()` when capture mode is active. Minimal changes, preserves lazy
optimization for non-capture users. Once applied, all `@warp_capturable(False)`
annotations for Tier 2 access can be removed and these terms become fully capturable.

Long-term: Migrate MDP kernels to use compound Tier 1 types directly. Best performance,
no derived property overhead at all.

# Articulation Performance Optimizations

**Date:** 2026-04-20
**Scope:** PhysX and Newton `Articulation` + `ArticulationData` classes
**Goal:** Reduce Python-side dispatch overhead in setter/writer methods

## Context

Micro-benchmarks (4096 instances, 12 bodies, 11 joints, RTX 5000 Ada) show that
setter/writer methods are dominated by Python dispatch overhead, not GPU kernel
execution. Fill-ratio experiments (5% vs 95% vs 100% of env_ids) show nearly
identical timings, confirming the bottleneck is above the kernel.

### Baseline Measurements (microseconds)

**PhysX Articulation — setter methods:**

| Method                          | torch_list | torch_tensor | tensor_5pct | tensor_100pct |
|---------------------------------|------------|--------------|-------------|---------------|
| `write_root_link_pose_to_sim`   | 180        | 94           | 108         | 112           |
| `write_joint_position_to_sim`   | 194        | 98           | 112         | 112           |
| `set_joint_position_target`     | 193        | 94           | 112         | 105           |
| `write_joint_stiffness_to_sim`  | 258        | 171          | 198         | 194           |
| `set_coms`                      | 436        | 350          | 384         | 366           |

**Newton Articulation — setter methods:**

| Method                          | torch_list | torch_tensor | warp_mask | tensor_5pct | tensor_100pct | mask_5pct | mask_100pct |
|---------------------------------|------------|--------------|-----------|-------------|---------------|-----------|-------------|
| `write_root_link_pose_to_sim`   | 194        | 100          | 73        | 117         | 120           | 82        | 81          |
| `write_joint_position_to_sim`   | 220        | 116          | 76        | 132         | 136           | 84        | 87          |
| `set_joint_position_target`     | 194        | 92           | 51        | 106         | 108           | 58        | 60          |
| `write_joint_stiffness_to_sim`  | 224        | 113          | 75        | 135         | 137           | 86        | 85          |
| `set_coms`                      | 223        | 118          | 75        | 140         | 142           | 89        | 89          |

**Key insight:** Fill ratio does not affect cost. The overhead is entirely in
resolve/dispatch/marshalling. The warp_mask path (Newton only) is consistently
fastest because it skips `_resolve_*` entirely.

### Cost Breakdown (approximate, per call)

| Component              | torch_list | torch_tensor | warp_mask |
|------------------------|------------|--------------|-----------|
| isinstance checks      | ~5 us      | ~5 us        | ~3 us     |
| list->warp alloc (CPU->GPU) | ~80 us | —           | —         |
| wp.from_torch metadata | —          | ~15 us       | —         |
| shape assertion        | ~5 us      | ~5 us        | ~5 us     |
| wp.launch overhead     | ~30 us     | ~30 us       | ~30 us    |
| GPU kernel             | <5 us      | <5 us        | <5 us     |
| PhysX CPU clone (joint props) | ~60 us | ~60 us    | N/A       |

## Optimizations

### 1. Remove Dead Kernel Arguments (Newton)

**Problem:** Newton kernels like `set_root_link_pose_to_sim_index` accept output
arrays for deprecated state buffers (`root_link_state_w`, `root_state_w`,
`root_com_state_w`) that are always passed as `None`. Warp still marshals these
arguments and emits null-check branches in the generated code.

**Affected kernels** (in `isaaclab_newton/assets/kernels.py`):

- `set_root_link_pose_to_sim_index` / `_mask` — 2 dead outputs
- `set_root_com_pose_to_sim_index` / `_mask` — 3 dead outputs
- `set_root_link_velocity_to_sim_index` / `_mask` — 3 dead outputs
- `set_root_com_velocity_to_sim_index` / `_mask` — 3 dead outputs

**Change:** Create slimmed kernels that only accept the required arguments.
Remove the `if root_link_state_w:` branches. Update all call sites to use the
new kernels.

**Expected impact:** ~5-10 us per call (fewer args to marshal, simpler codegen).

### 2. Cache `wp.from_torch` Results in `_resolve_*` Methods

**Problem:** `wp.from_torch` is zero-copy (same GPU memory) but allocates a new
Python `wp.array` wrapper each call. In RL training loops, callers often pass the
same tensor object (same `data_ptr()`) across steps.

**Change:** Add a single-slot cache to each resolve method, keyed on
`tensor.data_ptr()`. On cache hit, return the previously created warp array.

```python
# Sketch — both PhysX and Newton
def _resolve_env_ids(self, env_ids):
    if env_ids is None or env_ids == slice(None):
        return self._ALL_INDICES
    if isinstance(env_ids, torch.Tensor):
        ptr = env_ids.data_ptr()
        if self._cached_env_ids_ptr == ptr:
            return self._cached_env_ids_wp
        if env_ids.dtype == torch.int64:
            env_ids = env_ids.to(torch.int32)
        result = wp.from_torch(env_ids, dtype=wp.int32)
        self._cached_env_ids_ptr = ptr
        self._cached_env_ids_wp = result
        return result
    if isinstance(env_ids, list):
        return wp.array(env_ids, dtype=wp.int32, device=self.device)
    return env_ids
```

**Applies to:** Both PhysX and Newton — `_resolve_env_ids`,
`_resolve_joint_ids`, `_resolve_body_ids`.

**Cache fields per articulation instance:**

- `_cached_env_ids_ptr: int = -1`
- `_cached_env_ids_wp: wp.array | None = None`
- Same pattern for joint_ids and body_ids (6 fields total).

**Expected impact:** ~10-15 us per call when same tensor is reused (eliminates
`wp.array` Python object allocation).

### 3. Cache Warp `.view()` Results in PhysX Writers

**Problem:** PhysX writers call `.view(wp.float32)` on structured-typed warp
arrays every write call:

```python
self.root_view.set_root_transforms(
    self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids
)
```

The `.view()` creates a new `wp.array` wrapper pointing at the same memory.

**Change:** Pre-compute and cache the float32 view alongside the source buffer.
Store as `_root_link_pose_w_f32` (or similar). Invalidate/recreate when the
buffer pointer changes (after sim reset via `_create_simulation_bindings`).

**Cached views needed:**

- `_root_link_pose_w.data` -> `.view(wp.float32)` (used in root pose writers)
- `_root_com_vel_w.data` -> `.view(wp.float32)` (used in root velocity writers)
- `_root_link_vel_w.data` -> `.view(wp.float32)` (used in link velocity writers)
- `_body_com_pose_b.data` -> `.view(wp.transformf)` then `.view(wp.float32).reshape(...)` (used in set_coms)

**Applies to:** PhysX only (Newton kernels accept structured types natively).

**Expected impact:** ~3-5 us per call.

### 4. Pre-allocate Pinned CPU Buffers for PhysX Joint Property Writes

**Problem:** PhysX joint property writers (stiffness, damping, limits, armature,
friction, masses, coms, inertias) must send data to the PhysX TensorAPI on CPU.
Each call does two GPU->CPU copies that allocate fresh pageable CPU memory:

```python
cpu_env_ids = self._get_cpu_env_ids(env_ids)           # wp.clone(env_ids, device="cpu")
self.root_view.set_dof_stiffnesses(
    wp.clone(self.data._joint_stiffness, device="cpu"), # wp.clone(data, device="cpu")
    indices=cpu_env_ids
)
```

**Change:** Pre-allocate **pinned** CPU buffers at init time using
`wp.zeros(..., device="cpu", pinned=True)`. Use `wp.copy(dst, src)` instead of
`wp.clone`.

**Pinned buffers to pre-allocate:**

```python
# In _create_buffers:
self._cpu_env_ids_all = wp.zeros(self.num_instances, dtype=wp.int32, device="cpu", pinned=True)
wp.copy(self._cpu_env_ids_all, self._ALL_INDICES)

# Joint property buffers (only needed for "model" properties that require CPU writes):
self._cpu_joint_stiffness = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
self._cpu_joint_damping = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
self._cpu_joint_pos_limits = wp.zeros((N, J), dtype=wp.vec2f, device="cpu", pinned=True)
self._cpu_joint_vel_limits = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
self._cpu_joint_effort_limits = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
self._cpu_joint_armature = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
self._cpu_joint_friction_props = wp.zeros((N, J, 3), dtype=wp.float32, device="cpu", pinned=True)
self._cpu_body_mass = wp.zeros((N, B), dtype=wp.float32, device="cpu", pinned=True)
self._cpu_body_coms = wp.zeros((N, B, 7), dtype=wp.float32, device="cpu", pinned=True)
self._cpu_body_inertia = wp.zeros((N, B, 9), dtype=wp.float32, device="cpu", pinned=True)
```

**Usage pattern:**

```python
# Full-index write (hot path):
wp.copy(self._cpu_joint_stiffness, self.data._joint_stiffness)
self.root_view.set_dof_stiffnesses(self._cpu_joint_stiffness, indices=self._cpu_env_ids_all)

# Partial-index write (reset path — infrequent):
# Resets happen rarely and with variable-sized subsets, so a fresh
# wp.clone for the small env_ids array is acceptable. The data buffer
# is still copied into the full-sized pinned buffer (the PhysX view
# uses the indices arg to select which rows to apply).
cpu_env_ids = wp.clone(env_ids, device="cpu")
wp.copy(self._cpu_joint_stiffness, self.data._joint_stiffness)
self.root_view.set_dof_stiffnesses(self._cpu_joint_stiffness, indices=cpu_env_ids)
```

**Applies to:** PhysX only.

**Expected impact:** ~20-40 us per joint property write (eliminates CPU malloc +
enables DMA fast path via pinned memory).

## Files to Modify

### Newton

| File | Changes |
|------|---------|
| `isaaclab_newton/assets/kernels.py` | New slimmed kernel variants without dead output args |
| `isaaclab_newton/assets/articulation/articulation.py` | Update kernel call sites; add resolve caches |

### PhysX

| File | Changes |
|------|---------|
| `isaaclab_physx/assets/articulation/articulation.py` | Add resolve caches; cache `.view()` results; pre-allocate pinned CPU buffers |

### Shared (base class)

| File | Changes |
|------|---------|
| None | Resolve methods are backend-specific, not in the base class |

## Testing Strategy

1. **Benchmark comparison:** Re-run the existing benchmarks after each
   optimization and compare against the baseline numbers in this spec.
2. **Correctness:** Run existing articulation unit tests (`./isaaclab.sh -p -m
   pytest` on the relevant test directories) to verify no regressions.
3. **Cache correctness:** Add targeted tests that verify cached values are
   invalidated correctly after sim reset.

## Out of Scope

- Optimizing the list input path — lists are a legacy pattern and not worth
  caching (content verification is O(N)).
- Optimizing `wp.launch` overhead itself — that is warp-internal.
- Changing the public API signatures.

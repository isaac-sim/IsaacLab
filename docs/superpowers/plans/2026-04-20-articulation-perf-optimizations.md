# Articulation Performance Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Python-side dispatch overhead in PhysX and Newton articulation setter/writer methods by eliminating redundant allocations, dead kernel arguments, and GPU-CPU copies.

**Architecture:** Four independent optimizations applied to the articulation classes: (1) slim Newton kernels to remove dead state-buffer outputs, (2) cache `wp.from_torch` results in resolve methods for both backends, (3) cache `.view()` warp array wrappers in PhysX writers, (4) pre-allocate pinned CPU buffers for PhysX joint property writes.

**Tech Stack:** Python, Warp (`wp`), PyTorch, Isaac Lab benchmark framework

---

## Pre-requisite: Branch Setup

- [ ] **Step 0: Checkout develop and create feature branch**

```bash
git checkout develop
git pull origin develop
git checkout -b antoiner/perf/articulation-resolve-caching
```

---

### Task 1: Remove Dead Kernel Arguments (Newton)

Newton kernels accept output arrays for deprecated state buffers that are always passed as `None`. Removing them reduces arg marshalling and simplifies warp codegen.

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/kernels.py:471-818`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py` (8 call sites)
- Test: `source/isaaclab_newton/test/assets/test_articulation.py`

- [ ] **Step 1: Slim `set_root_link_pose_to_sim_index` and `_mask` kernels**

In `source/isaaclab_newton/isaaclab_newton/assets/kernels.py`, replace the existing kernels at lines 471-528 with slimmed versions that remove the `root_link_state_w` and `root_state_w` parameters:

```python
@wp.kernel
def set_root_link_pose_to_sim_index(
    data: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root link pose data to simulation buffers.

    Args:
        data: Input array of root link poses. Shape is (num_selected_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        root_link_pose_w: Output array where root link poses are written. Shape is (num_envs,).
    """
    i = wp.tid()
    root_link_pose_w[env_ids[i]] = data[i]


@wp.kernel
def set_root_link_pose_to_sim_mask(
    data: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root link pose data to simulation buffers using a mask.

    Args:
        data: Input array of root link poses. Shape is (num_instances,).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        root_link_pose_w: Output array where root link poses are written. Shape is (num_envs,).
    """
    i = wp.tid()
    if env_mask[i]:
        root_link_pose_w[i] = data[i]
```

- [ ] **Step 2: Slim `set_root_com_pose_to_sim_index` and `_mask` kernels**

Replace lines 531-621 — remove `root_com_state_w`, `root_link_state_w`, and `root_state_w` parameters. Keep the required `root_com_pose_w`, `root_link_pose_w`, `body_com_pos_b` since those are live outputs:

```python
@wp.kernel
def set_root_com_pose_to_sim_index(
    data: wp.array(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root COM pose data to simulation buffers.

    Args:
        data: Input array of root COM poses. Shape is (num_selected_envs,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        root_com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
        root_link_pose_w: Output array where root link poses (derived from COM) are written.
            Shape is (num_envs,).
    """
    i = wp.tid()
    root_com_pose_w[env_ids[i]] = data[i]
    root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(
        root_com_pose_w[env_ids[i]], body_com_pos_b[env_ids[i], 0]
    )


@wp.kernel
def set_root_com_pose_to_sim_mask(
    data: wp.array(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    env_mask: wp.array(dtype=wp.bool),
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root COM pose data to simulation buffers using a mask.

    Args:
        data: Input array of root COM poses. Shape is (num_instances,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        env_mask: Input array of environment mask. Shape is (num_instances,).
        root_com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
        root_link_pose_w: Output array where root link poses (derived from COM) are written.
            Shape is (num_envs,).
    """
    i = wp.tid()
    if env_mask[i]:
        root_com_pose_w[i] = data[i]
        root_link_pose_w[i] = get_com_pose_in_link_frame_func(root_com_pose_w[i], body_com_pos_b[i, 0])
```

- [ ] **Step 3: Slim `set_root_com_velocity_to_sim_index` and `_mask` kernels**

Replace lines 624-700 — remove `root_state_w` and `root_com_state_w` parameters. Keep `root_com_velocity_w` and `body_acc_w` (live outputs):

```python
@wp.kernel
def set_root_com_velocity_to_sim_index(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    num_bodies: wp.int32,
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root COM velocity data to simulation buffers.

    Args:
        data: Input array of root COM spatial velocities. Shape is (num_selected_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        num_bodies: Input scalar number of bodies per environment.
        root_com_velocity_w: Output array where root COM velocities are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is (num_envs, num_bodies).
    """
    i = wp.tid()
    root_com_velocity_w[env_ids[i]] = data[i]
    for j in range(num_bodies):
        body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_root_com_velocity_to_sim_mask(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32,
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root COM velocity data to simulation buffers using a mask.

    Args:
        data: Input array of root COM spatial velocities. Shape is (num_instances,).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        num_bodies: Input scalar number of bodies per environment.
        root_com_velocity_w: Output array where root COM velocities are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is (num_envs, num_bodies).
    """
    i = wp.tid()
    if env_mask[i]:
        root_com_velocity_w[i] = data[i]
        for j in range(num_bodies):
            body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
```

- [ ] **Step 4: Slim `set_root_link_velocity_to_sim_index` and `_mask` kernels**

Replace lines 703-817 — remove `root_link_state_w`, `root_state_w`, and `root_com_state_w` parameters. Keep `root_link_velocity_w`, `root_com_velocity_w`, `body_acc_w`, `body_com_pos_b`, `link_pose_w` (live inputs/outputs):

```python
@wp.kernel
def set_root_link_velocity_to_sim_index(
    data: wp.array(dtype=wp.spatial_vectorf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    link_pose_w: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    num_bodies: wp.int32,
    root_link_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root link velocity data to simulation buffers.

    Args:
        data: Input array of root link spatial velocities. Shape is (num_selected_envs,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        link_pose_w: Input array of root link poses in world frame. Shape is (num_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        num_bodies: Input scalar number of bodies per environment.
        root_link_velocity_w: Output array where root link velocities are written. Shape is (num_envs,).
        root_com_velocity_w: Output array where root COM velocities (derived from link) are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is (num_envs, num_bodies).
    """
    i = wp.tid()
    root_link_velocity_w[env_ids[i]] = data[i]
    root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(
        root_link_velocity_w[env_ids[i]], link_pose_w[env_ids[i]], body_com_pos_b[env_ids[i], 0]
    )
    for j in range(num_bodies):
        body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_root_link_velocity_to_sim_mask(
    data: wp.array(dtype=wp.spatial_vectorf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    link_pose_w: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32,
    root_link_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root link velocity data to simulation buffers using a mask.

    Args:
        data: Input array of root link spatial velocities. Shape is (num_instances,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        link_pose_w: Input array of root link poses in world frame. Shape is (num_envs,).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        num_bodies: Input scalar number of bodies per environment.
        root_link_velocity_w: Output array where root link velocities are written. Shape is (num_envs,).
        root_com_velocity_w: Output array where root COM velocities (derived from link) are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is (num_envs, num_bodies).
    """
    i = wp.tid()
    if env_mask[i]:
        root_link_velocity_w[i] = data[i]
        root_com_velocity_w[i] = get_link_velocity_in_com_frame_func(
            root_link_velocity_w[i], link_pose_w[i], body_com_pos_b[i, 0]
        )
        for j in range(num_bodies):
            body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
```

- [ ] **Step 5: Update call sites in Newton `articulation.py`**

Update all 8 `wp.launch` call sites in `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py` to remove the `None` outputs. For each affected method, remove the dead `None` entries from the `outputs` list.

The 8 call sites (by line number) and their dead args to remove:

1. **Line 478** (`write_root_link_pose_to_sim_index`): Remove `None, None` from outputs → `outputs=[self.data.root_link_pose_w]`
2. **Line 534** (`write_root_link_pose_to_sim_mask`): Remove `None, None` from outputs → `outputs=[self.data.root_link_pose_w]`
3. **Line 594** (`write_root_com_pose_to_sim_index`): Remove `None, None, None` from outputs → `outputs=[self.data._root_com_pose_w.data, self.data.root_link_pose_w]`
4. **Line 655** (`write_root_com_pose_to_sim_mask`): Remove `None, None, None` from outputs → `outputs=[self.data._root_com_pose_w.data, self.data.root_link_pose_w]`
5. **Line 771** (`write_root_com_velocity_to_sim_index`): Remove `None, None` from outputs → `outputs=[self.data._sim_bind_root_com_vel_w, self.data._body_com_acc_w.data]`
6. **Line 819** (`write_root_com_velocity_to_sim_mask`): Remove `None, None` from outputs → `outputs=[self.data._sim_bind_root_com_vel_w, self.data._body_com_acc_w.data]`
7. **Line 870** (`write_root_link_velocity_to_sim_index`): Remove `None, None, None` from outputs → `outputs=[self.data._root_link_vel_w.data, self.data._sim_bind_root_com_vel_w, self.data._body_com_acc_w.data]`
8. **Line 924** (`write_root_link_velocity_to_sim_mask`): Remove `None, None, None` from outputs → `outputs=[self.data._root_link_vel_w.data, self.data._sim_bind_root_com_vel_w, self.data._body_com_acc_w.data]`

- [ ] **Step 6: Run Newton articulation tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/assets/test_articulation.py -v
```

Expected: All tests pass.

- [ ] **Step 7: Run Newton benchmark and compare**

```bash
./isaaclab.sh -p source/isaaclab_newton/benchmark/assets/benchmark_articulation.py --headless --mode warp_mask 2>&1 | grep -E '^\[|^Bench'
```

Compare `warp_mask` numbers against baseline (e.g., `write_root_link_pose_to_sim_mask` baseline: 73 us).

- [ ] **Step 8: Commit**

```bash
git add source/isaaclab_newton/isaaclab_newton/assets/kernels.py source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py
git commit -m "Remove dead state-buffer outputs from Newton root pose/velocity kernels

Slim 8 warp kernels (index + mask variants for root link pose, root COM
pose, root COM velocity, root link velocity) to remove output parameters
that are always passed as None. Reduces arg marshalling overhead and
simplifies generated GPU code."
```

---

### Task 2: Cache `wp.from_torch` Results in Resolve Methods (Both Backends)

Add single-slot caches to `_resolve_env_ids`, `_resolve_joint_ids`, and `_resolve_body_ids` so that repeated calls with the same torch tensor skip the `wp.from_torch` wrapper allocation.

**Files:**
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:3646-3680,4259-4347`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py:3230-3242,3678-3732`

- [ ] **Step 1: Add cache fields to PhysX `_create_buffers`**

In `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`, at the end of the `_create_buffers` method (after line ~3653), add:

```python
        # Single-slot caches for _resolve_* methods (keyed on tensor.data_ptr())
        self._cached_env_ids_ptr: int = -1
        self._cached_env_ids_wp: wp.array | None = None
        self._cached_joint_ids_ptr: int = -1
        self._cached_joint_ids_wp: wp.array | None = None
        self._cached_body_ids_ptr: int = -1
        self._cached_body_ids_wp: wp.array | None = None
```

- [ ] **Step 2: Update PhysX `_resolve_env_ids`**

Replace the method at line 4259 with:

```python
    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        """Resolve environment indices to a warp array.

        Uses a single-slot cache to avoid repeated ``wp.from_torch`` wrapper
        allocations when the same tensor is passed across steps.

        Args:
            env_ids: Environment indices. If None, then all indices are used.

        Returns:
            A warp array of environment indices.
        """
        if (env_ids is None) or (env_ids == slice(None)):
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

- [ ] **Step 3: Update PhysX `_resolve_joint_ids`**

Replace the method at line 4299 with:

```python
    def _resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array | torch.Tensor:
        """Resolve joint indices to a warp array or tensor.

        Uses a single-slot cache to avoid repeated ``wp.from_torch`` wrapper
        allocations when the same tensor is passed across steps.

        Args:
            joint_ids: Joint indices. If None, then all indices are used.

        Returns:
            A warp array of joint indices or a tensor of joint indices.
        """
        if isinstance(joint_ids, list):
            return wp.array(joint_ids, dtype=wp.int32, device=self.device)
        if (joint_ids is None) or (joint_ids == slice(None)):
            return self._ALL_JOINT_INDICES
        if isinstance(joint_ids, torch.Tensor):
            ptr = joint_ids.data_ptr()
            if self._cached_joint_ids_ptr == ptr:
                return self._cached_joint_ids_wp
            result = wp.from_torch(joint_ids, dtype=wp.int32)
            self._cached_joint_ids_ptr = ptr
            self._cached_joint_ids_wp = result
            return result
        return joint_ids
```

- [ ] **Step 4: Update PhysX `_resolve_body_ids`**

Replace the method at line 4334 with:

```python
    def _resolve_body_ids(self, body_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array | torch.Tensor:
        """Resolve body indices to a warp array or tensor.

        Uses a single-slot cache to avoid repeated ``wp.from_torch`` wrapper
        allocations when the same tensor is passed across steps.

        Args:
            body_ids: Body indices. If None, then all indices are used.

        Returns:
            A warp array of body indices or a tensor of body indices.
        """
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self.device)
        if (body_ids is None) or (body_ids == slice(None)):
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, torch.Tensor):
            ptr = body_ids.data_ptr()
            if self._cached_body_ids_ptr == ptr:
                return self._cached_body_ids_wp
            result = wp.from_torch(body_ids, dtype=wp.int32)
            self._cached_body_ids_ptr = ptr
            self._cached_body_ids_wp = result
            return result
        return body_ids
```

- [ ] **Step 5: Add cache fields to Newton `_create_buffers`**

In `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`, at the end of the `_create_buffers` method (after line ~3242), add the same 6 cache fields:

```python
        # Single-slot caches for _resolve_* methods (keyed on tensor.data_ptr())
        self._cached_env_ids_ptr: int = -1
        self._cached_env_ids_wp: wp.array | None = None
        self._cached_joint_ids_ptr: int = -1
        self._cached_joint_ids_wp: wp.array | None = None
        self._cached_body_ids_ptr: int = -1
        self._cached_body_ids_wp: wp.array | None = None
```

- [ ] **Step 6: Update Newton `_resolve_env_ids`**

Replace the method at line 3678 with the same cached version as PhysX Step 2 (identical logic).

- [ ] **Step 7: Update Newton `_resolve_joint_ids`**

Replace the method at line 3701 with the same cached version as PhysX Step 3 (identical logic).

- [ ] **Step 8: Update Newton `_resolve_body_ids`**

Replace the method at line 3719 with the same cached version as PhysX Step 4 (identical logic).

- [ ] **Step 9: Run tests for both backends**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py -v
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/assets/test_articulation.py -v
```

Expected: All tests pass.

- [ ] **Step 10: Run benchmarks and compare**

```bash
# PhysX
./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation.py --headless --mode torch_tensor 2>&1 | grep -E '^\[|^Bench'
# Newton
./isaaclab.sh -p source/isaaclab_newton/benchmark/assets/benchmark_articulation.py --headless --mode torch_tensor 2>&1 | grep -E '^\[|^Bench'
```

Compare `torch_tensor` numbers against baseline (e.g., PhysX `set_joint_position_target`: baseline 94 us).

- [ ] **Step 11: Commit**

```bash
git add source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py
git commit -m "Cache wp.from_torch results in _resolve_* methods

Add single-slot caches keyed on tensor.data_ptr() to _resolve_env_ids,
_resolve_joint_ids, and _resolve_body_ids in both PhysX and Newton
backends. Eliminates redundant wp.array wrapper allocations when the
same tensor is passed across training steps."
```

---

### Task 3: Cache Warp `.view()` Results in PhysX Writers

PhysX writers call `.view(wp.float32)` on structured warp arrays every call, creating a new wrapper object each time. Cache these views.

**Files:**
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:3646-3680` (add cached views to `_create_buffers`)
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:478,570,709,802,2108-2113` (use cached views)

- [ ] **Step 1: Add cached view fields to `_create_buffers`**

In `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`, at the end of `_create_buffers`, add:

```python
        # Cached .view(wp.float32) wrappers for structured warp arrays.
        # These avoid per-call wp.array metadata allocation in writers.
        # Recreated in _create_simulation_bindings after sim reset.
        self._root_link_pose_w_f32: wp.array | None = None
        self._root_com_vel_w_f32: wp.array | None = None
        self._root_link_vel_w_f32: wp.array | None = None
```

- [ ] **Step 2: Add a helper method to get or create cached views**

Add this method to the PhysX `Articulation` class (near the resolve methods):

```python
    def _get_root_link_pose_w_f32(self) -> wp.array:
        """Get a cached float32 view of root_link_pose_w for PhysX TensorAPI."""
        if self._root_link_pose_w_f32 is None:
            self._root_link_pose_w_f32 = self.data._root_link_pose_w.data.view(wp.float32)
        return self._root_link_pose_w_f32

    def _get_root_com_vel_w_f32(self) -> wp.array:
        """Get a cached float32 view of root_com_vel_w for PhysX TensorAPI."""
        if self._root_com_vel_w_f32 is None:
            self._root_com_vel_w_f32 = self.data._root_com_vel_w.data.view(wp.float32)
        return self._root_com_vel_w_f32

    def _get_root_link_vel_w_f32(self) -> wp.array:
        """Get a cached float32 view of root_link_vel_w for PhysX TensorAPI."""
        if self._root_link_vel_w_f32 is None:
            self._root_link_vel_w_f32 = self.data._root_link_vel_w.data.view(wp.float32)
        return self._root_link_vel_w_f32
```

- [ ] **Step 3: Update call sites to use cached views**

Replace the `.view(wp.float32)` calls at the following lines:

Line 478 — `write_root_link_pose_to_sim_index`:
```python
# Before:
self.root_view.set_root_transforms(self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids)
# After:
self.root_view.set_root_transforms(self._get_root_link_pose_w_f32(), indices=env_ids)
```

Line 570 — `write_root_link_pose_to_sim_mask` (same replacement).

Line 709 — `write_root_com_velocity_to_sim_index`:
```python
# Before:
self.root_view.set_root_velocities(self.data._root_com_vel_w.data.view(wp.float32), indices=env_ids)
# After:
self.root_view.set_root_velocities(self._get_root_com_vel_w_f32(), indices=env_ids)
```

Line 802 — `write_root_link_velocity_to_sim_index`:
```python
# Before:
self.root_view.set_root_velocities(self.data._root_link_vel_w.data.view(wp.float32), indices=env_ids)
# After:
self.root_view.set_root_velocities(self._get_root_link_vel_w_f32(), indices=env_ids)
```

- [ ] **Step 4: Run PhysX tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py
git commit -m "Cache warp .view(wp.float32) results in PhysX writers

Pre-compute and cache float32 views of structured warp arrays used by
PhysX TensorAPI calls. Eliminates per-call wp.array wrapper allocation
in root pose and velocity writers."
```

---

### Task 4: Pre-allocate Pinned CPU Buffers for PhysX Joint Property Writes

PhysX joint property writers do GPU->CPU copies with `wp.clone(device="cpu")` each call, allocating fresh pageable CPU memory. Pre-allocate pinned buffers and use `wp.copy` instead.

**Files:**
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:3646-3680` (add pinned buffers to `_create_buffers`)
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:4226-4238` (update `_get_cpu_env_ids`)
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py` (12+ writer methods)

- [ ] **Step 1: Add pinned CPU buffer allocations to `_create_buffers`**

In `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`, at the end of `_create_buffers`, add:

```python
        # Pre-allocated pinned CPU buffers for PhysX TensorAPI writes.
        # PhysX requires CPU arrays for "model" property updates (stiffness, damping, etc.).
        # Pinned memory enables DMA fast path and avoids per-call malloc.
        N, J, B = self.num_instances, self.num_joints, self.num_bodies
        self._cpu_env_ids_all = wp.zeros(N, dtype=wp.int32, device="cpu", pinned=True)
        wp.copy(self._cpu_env_ids_all, self._ALL_INDICES)
        self._cpu_joint_stiffness = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_joint_damping = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_joint_pos_limits = wp.zeros((N, J, 2), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_joint_vel_limits = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_joint_effort_limits = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_joint_armature = wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_joint_friction_props = wp.zeros((N, J, 3), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_body_mass = wp.zeros((N, B), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_body_coms = wp.zeros((N, B, 7), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_body_inertia = wp.zeros((N, B, 9), dtype=wp.float32, device="cpu", pinned=True)
```

- [ ] **Step 2: Update `_get_cpu_env_ids` to use pinned buffer for full-index case**

Replace the method at line 4226 with:

```python
    def _get_cpu_env_ids(self, env_ids: wp.array | torch.Tensor) -> wp.array:
        """Get the CPU environment indices.

        For the full-index case (all environments), returns the pre-allocated
        pinned CPU buffer. For partial indices, clones to CPU (infrequent path).

        Args:
            env_ids: Environment indices.

        Returns:
            A warp array of environment indices on CPU.
        """
        if isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids, dtype=wp.int32)
        # Fast path: if these are all indices, use pre-allocated pinned buffer
        if env_ids.ptr == self._ALL_INDICES.ptr:
            return self._cpu_env_ids_all
        # Slow path: partial indices (reset), clone to CPU
        return wp.clone(env_ids, device="cpu")
```

- [ ] **Step 3: Update joint stiffness writer to use pinned buffer**

At line 1088-1089, replace:

```python
# Before:
cpu_env_ids = self._get_cpu_env_ids(env_ids)
self.root_view.set_dof_stiffnesses(wp.clone(self.data._joint_stiffness, device="cpu"), indices=cpu_env_ids)

# After:
cpu_env_ids = self._get_cpu_env_ids(env_ids)
wp.copy(self._cpu_joint_stiffness, self.data._joint_stiffness)
self.root_view.set_dof_stiffnesses(self._cpu_joint_stiffness, indices=cpu_env_ids)
```

- [ ] **Step 4: Update remaining joint property writers**

Apply the same pattern (replace `wp.clone(..., device="cpu")` with `wp.copy` into pre-allocated pinned buffer) to all remaining writers. Each writer's data buffer and pinned buffer:

| Writer method | Data source | Pinned buffer |
|---|---|---|
| `write_joint_damping_to_sim_index` (line ~1183) | `self.data._joint_damping` | `self._cpu_joint_damping` |
| `write_joint_position_limit_to_sim_index` (line ~1283) | `self.data._joint_pos_limits` | `self._cpu_joint_pos_limits` |
| `write_joint_velocity_limit_to_sim_index` (line ~1387) | `self.data._joint_vel_limits` | `self._cpu_joint_vel_limits` |
| `write_joint_effort_limit_to_sim_index` (line ~1488) | `self.data._joint_effort_limits` | `self._cpu_joint_effort_limits` |
| `write_joint_armature_to_sim_index` (line ~1587) | `self.data._joint_armature` | `self._cpu_joint_armature` |
| `write_joint_friction_coefficient_to_sim_index` (line ~1726) | `friction_props` | `self._cpu_joint_friction_props` |
| `set_masses_index` (line ~2030) | `self.data._body_mass` | `self._cpu_body_mass` |
| `set_coms_index` (line ~2108-2114) | `self.data._body_com_pose_b.data` | `self._cpu_body_coms` |
| `set_inertias_index` (line ~2192) | `self.data._body_inertia` | `self._cpu_body_inertia` |

For each, the pattern is:
```python
# Before:
cpu_env_ids = self._get_cpu_env_ids(env_ids)
self.root_view.set_dof_XXX(wp.clone(self.data._XXX, device="cpu"), indices=cpu_env_ids)

# After:
cpu_env_ids = self._get_cpu_env_ids(env_ids)
wp.copy(self._cpu_XXX, self.data._XXX)
self.root_view.set_dof_XXX(self._cpu_XXX, indices=cpu_env_ids)
```

Note: `set_coms_index` has a special pattern with `.view(wp.float32).reshape(...)`. Replace:
```python
# Before:
body_com_flat = (
    wp.clone(self.data._body_com_pose_b.data, device="cpu")
    .view(wp.float32)
    .reshape((self.num_instances, self.num_bodies, 7))
)
# After:
wp.copy(self._cpu_body_coms, self.data._body_com_pose_b.data.view(wp.float32).reshape((self.num_instances, self.num_bodies, 7)))
body_com_flat = self._cpu_body_coms
```

- [ ] **Step 5: Update `_mask` variants**

The `_mask` variants of these writers follow the same pattern (they also call `_get_cpu_env_ids` and `wp.clone`). Apply identical pinned-buffer replacements to all `_mask` writer methods.

- [ ] **Step 6: Run PhysX tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py -v
```

Expected: All tests pass.

- [ ] **Step 7: Run PhysX benchmark and compare**

```bash
./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation.py --headless 2>&1 | grep -E '^\[|^Bench|^==='
```

Compare joint property write numbers against baseline (e.g., `write_joint_stiffness_to_sim` torch_tensor baseline: 171 us).

- [ ] **Step 8: Commit**

```bash
git add source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py
git commit -m "Pre-allocate pinned CPU buffers for PhysX joint property writes

Replace per-call wp.clone(device='cpu') with wp.copy into pre-allocated
pinned CPU buffers for all joint property and body property writers.
Pinned memory enables DMA fast path and eliminates CPU malloc overhead.
Full-index env_ids also use a pre-allocated pinned buffer."
```

---

### Task 5: Final Benchmark Comparison

Run full benchmarks for both backends and compare against the baseline in the spec.

**Files:**
- Run: `source/isaaclab_physx/benchmark/assets/benchmark_articulation.py`
- Run: `source/isaaclab_newton/benchmark/assets/benchmark_articulation.py`
- Run: `source/isaaclab_physx/benchmark/assets/benchmark_articulation_data.py`
- Run: `source/isaaclab_newton/benchmark/assets/benchmark_articulation_data.py`

- [ ] **Step 1: Run PhysX full benchmark**

```bash
./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation.py --headless 2>&1 | grep -E '^\[|^Bench|^==='
```

- [ ] **Step 2: Run Newton full benchmark**

```bash
./isaaclab.sh -p source/isaaclab_newton/benchmark/assets/benchmark_articulation.py --headless 2>&1 | grep -E '^\[|^Bench|^==='
```

- [ ] **Step 3: Run data benchmarks (regression check)**

```bash
./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation_data.py --headless 2>&1 | grep -E '^\[|^Bench'
./isaaclab.sh -p source/isaaclab_newton/benchmark/assets/benchmark_articulation_data.py --headless 2>&1 | grep -E '^\[|^Bench'
```

- [ ] **Step 4: Compare results against baseline and document improvements**

Create a summary comparing before/after numbers for each optimization.

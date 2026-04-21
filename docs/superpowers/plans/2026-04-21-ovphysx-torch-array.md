# OVPhysX TorchArray Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the OVPhysX `ArticulationData` class from returning raw `wp.array` to returning `TorchArray`, matching the pattern already established in PhysX and Newton backends.

**Architecture:** Every property getter that currently returns `wp.array` will be changed to return `TorchArray` via a lazy `_ta` cache pattern. A `TorchArray` is initialized as `None` in `_create_buffers()`, checked for `None` on first property access, and then cached for subsequent calls. For `TimestampedBuffer`-backed properties, the `TorchArray` wraps `buf.data`. For plain `wp.array` fields, it wraps the field directly. `GRAVITY_VEC_W` and `FORWARD_VEC_B` are wrapped at construction time.

**Tech Stack:** Python, warp, `TorchArray` from `isaaclab.utils.warp`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py` | Modify | Change all 84 `-> wp.array` return types to `-> TorchArray`, add `_ta` cache fields in `_create_buffers()`, wrap returns in TorchArray |
| `source/isaaclab_ovphysx/test/assets/test_articulation_data.py` | Modify | Update existing test to use `.warp` when accessing raw warp data |
| `source/isaaclab_ovphysx/docs/CHANGELOG.rst` | Modify | Add changelog entry for TorchArray migration |
| `source/isaaclab_ovphysx/config/extension.toml` | Modify | Bump version to 0.1.1 |

## Reference: The TorchArray wrapping patterns

There are **four patterns** used across PhysX/Newton, depending on what backs the property:

### Pattern A: TimestampedBuffer property (read from binding then return)
```python
# BEFORE (OVPhysX today)
@property
def root_link_pose_w(self) -> wp.array:
    self._read_transform_binding(TT.ROOT_POSE, self._root_link_pose_w)
    return self._root_link_pose_w.data

# AFTER
@property
def root_link_pose_w(self) -> TorchArray:
    self._read_transform_binding(TT.ROOT_POSE, self._root_link_pose_w)
    if self._root_link_pose_w_ta is None:
        self._root_link_pose_w_ta = TorchArray(self._root_link_pose_w.data)
    return self._root_link_pose_w_ta
```

### Pattern B: TimestampedBuffer property (computed via wp.launch, timestamp-gated)
```python
# BEFORE
@property
def projected_gravity_b(self) -> wp.array:
    if self._projected_gravity_b.timestamp < self._sim_timestamp:
        wp.launch(...)
        self._projected_gravity_b.timestamp = self._sim_timestamp
    return self._projected_gravity_b.data

# AFTER
@property
def projected_gravity_b(self) -> TorchArray:
    if self._projected_gravity_b.timestamp < self._sim_timestamp:
        wp.launch(...)
        self._projected_gravity_b.timestamp = self._sim_timestamp
    if self._projected_gravity_b_ta is None:
        self._projected_gravity_b_ta = TorchArray(self._projected_gravity_b.data)
    return self._projected_gravity_b_ta
```

### Pattern C: Plain wp.array field (static property)
```python
# BEFORE
@property
def body_mass(self) -> wp.array:
    return self._body_mass

# AFTER
@property
def body_mass(self) -> TorchArray:
    if self._body_mass_ta is None:
        self._body_mass_ta = TorchArray(self._body_mass)
    return self._body_mass_ta
```

### Pattern D: Constructor constant
```python
# BEFORE
self.GRAVITY_VEC_W = wp.from_numpy(gravity_dir_tiled, dtype=wp.vec3f, device=device)

# AFTER
self.GRAVITY_VEC_W = TorchArray(wp.from_numpy(gravity_dir_tiled, dtype=wp.vec3f, device=device))
```

### Identifying which pattern to use

- If the property calls `self._read_*_binding(...)` then reads `buf.data` -> **Pattern A**
- If the property checks `buf.timestamp < self._sim_timestamp` and calls `wp.launch(...)` -> **Pattern B**
- If the property just returns `self._some_field` (a plain `wp.array`) -> **Pattern C**
- If it's `GRAVITY_VEC_W` or `FORWARD_VEC_B` in `__init__` -> **Pattern D**

### Internal warp kernel arguments

When a property's `wp.array` is passed as an input to `wp.launch(...)` inside another property (e.g. `self.root_link_pose_w` passed into a kernel), call it directly — `TorchArray` exposes `__cuda_array_interface__` so `wp.launch` handles it transparently. No `.warp` unwrapping needed in kernel call sites.

---

### Task 1: Add TorchArray import and wrap GRAVITY_VEC_W / FORWARD_VEC_B

**Files:**
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py:14,106-107`

- [ ] **Step 1: Add TorchArray import**

At line 14 (after `import warp as wp`), add:

```python
from isaaclab.utils.warp import TorchArray
```

- [ ] **Step 2: Wrap GRAVITY_VEC_W and FORWARD_VEC_B in __init__**

Change lines 106-107 from:
```python
self.GRAVITY_VEC_W = wp.from_numpy(gravity_dir_tiled, dtype=wp.vec3f, device=device)
self.FORWARD_VEC_B = wp.from_numpy(forward_tiled, dtype=wp.vec3f, device=device)
```
to:
```python
self.GRAVITY_VEC_W = TorchArray(wp.from_numpy(gravity_dir_tiled, dtype=wp.vec3f, device=device))
self.FORWARD_VEC_B = TorchArray(wp.from_numpy(forward_tiled, dtype=wp.vec3f, device=device))
```

- [ ] **Step 3: Verify existing test still passes**

Run: `docker exec isaac-lab-base bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/assets/test_articulation_data.py -v"`

The test calls `data.joint_acc.numpy()` — this will break once `joint_acc` returns `TorchArray` (no `.numpy()` method). This is expected; we'll fix it in Task 5.

- [ ] **Step 4: Commit**

```
git add source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py
git commit -m "Add TorchArray import and wrap OVPhysX gravity/forward constants"
```

---

### Task 2: Add _ta cache fields to _create_buffers()

**Files:**
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py:1176-1285`

- [ ] **Step 1: Add `_ta = None` initialization for every TimestampedBuffer and plain wp.array field**

At the end of `_create_buffers()` (before the `self._read_initial_properties()` call at line 1285), add a block initializing all `_ta` cache fields to `None`. There are approximately 40+ fields that need `_ta` caches.

Group them to match the existing buffer sections:

```python
# -- TorchArray caches (initialized lazily on first property access)
# Root state
self._root_link_pose_w_ta = None
self._root_link_vel_w_ta = None
self._root_com_pose_w_ta = None
self._root_com_vel_w_ta = None

# Body state
self._body_link_pose_w_ta = None
self._body_link_vel_w_ta = None
self._body_com_pose_b_ta = None
self._body_com_pose_w_ta = None
self._body_com_vel_w_ta = None
self._body_com_acc_w_ta = None
self._body_incoming_joint_wrench_buf_ta = None

# Joint state
self._joint_pos_buf_ta = None
self._joint_vel_buf_ta = None
self._joint_acc_ta = None

# Joint properties
self._joint_stiffness_ta = None
self._joint_damping_ta = None
self._joint_armature_ta = None
self._joint_friction_coeff_ta = None
self._joint_pos_limits_ta = None
self._joint_vel_limits_ta = None
self._joint_effort_limits_ta = None

# Body properties
self._body_mass_ta = None
self._body_inertia_ta = None

# Soft limits / custom
self._soft_joint_pos_limits_ta = None
self._soft_joint_vel_limits_ta = None
self._gear_ratio_ta = None

# Command buffers
self._joint_pos_target_ta = None
self._joint_vel_target_ta = None
self._joint_effort_target_ta = None
self._computed_torque_ta = None
self._applied_torque_ta = None

# Default state
self._default_root_pose_ta = None
self._default_root_vel_ta = None
self._default_joint_pos_ta = None
self._default_joint_vel_ta = None

# Derived properties
self._projected_gravity_b_ta = None
self._heading_w_ta = None
self._root_link_lin_vel_b_ta = None
self._root_link_ang_vel_b_ta = None
self._root_com_lin_vel_b_ta = None
self._root_com_ang_vel_b_ta = None

# Tendon properties
self._fixed_tendon_stiffness_ta = None
self._fixed_tendon_damping_ta = None
self._fixed_tendon_limit_stiffness_ta = None
self._fixed_tendon_rest_length_ta = None
self._fixed_tendon_offset_ta = None
self._fixed_tendon_pos_limits_ta = None
self._spatial_tendon_stiffness_ta = None
self._spatial_tendon_damping_ta = None
self._spatial_tendon_limit_stiffness_ta = None
self._spatial_tendon_offset_ta = None
```

- [ ] **Step 2: Commit**

```
git add source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py
git commit -m "Add TorchArray cache fields to OVPhysX _create_buffers"
```

---

### Task 3: Convert all property return types to TorchArray

**Files:**
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py` (all ~84 property getters)

This is the bulk of the work. For each property:
1. Change `-> wp.array` to `-> TorchArray`
2. Before the `return` statement, add the `_ta is None` check + TorchArray wrapping
3. Return `self._foo_ta` instead of the raw warp array

Apply the correct pattern (A, B, C, or D) per property as documented above.

**Key properties to convert (non-exhaustive, grouped by section):**

Defaults: `default_root_pose`, `default_root_vel`, `default_joint_pos`, `default_joint_vel`
Commands: `joint_pos_target`, `joint_vel_target`, `joint_effort_target`
Actuator: `computed_torque`, `applied_torque`
Joint properties: `joint_stiffness`, `joint_damping`, `joint_armature`, `joint_friction_coeff`, `joint_pos_limits`, `joint_vel_limits`, `joint_effort_limits`
Soft limits: `soft_joint_pos_limits`, `soft_joint_vel_limits`, `gear_ratio`
Tendons: `fixed_tendon_stiffness`, `fixed_tendon_damping`, `fixed_tendon_limit_stiffness`, `fixed_tendon_rest_length`, `fixed_tendon_offset`, `fixed_tendon_pos_limits`, `spatial_tendon_stiffness`, `spatial_tendon_damping`, `spatial_tendon_limit_stiffness`, `spatial_tendon_offset`
Root state: `root_link_pose_w`, `root_link_vel_w`, `root_com_pose_w`, `root_com_vel_w`
Body state: `body_mass`, `body_inertia`, `body_link_pose_w`, `body_link_vel_w`, `body_com_pose_b`, `body_com_pose_w`, `body_com_vel_w`, `body_com_acc_w`
Derived: `projected_gravity_b`, `heading_w`, `root_link_lin_vel_b`, `root_link_ang_vel_b`, `root_com_lin_vel_b`, `root_com_ang_vel_b`
Joint state: `joint_pos`, `joint_vel`, `joint_acc`
View helpers: `root_pos_w`, `root_quat_w`, `root_lin_vel_w`, `root_ang_vel_w`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `body_link_quat_w`
Wrench/force: `body_incoming_wrench`

**Important:** Setter methods (e.g. `default_root_pose.setter`) should keep accepting `wp.array` — they write to the underlying buffer, not to TorchArray. No change needed for setters.

- [ ] **Step 1: Convert all property getters**

Work through all properties systematically, applying the correct pattern per property.

- [ ] **Step 2: Commit**

```
git add source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py
git commit -m "Convert OVPhysX ArticulationData properties to return TorchArray"
```

---

### Task 4: Handle view-based properties (root_pos_w, root_quat_w, etc.)

**Files:**
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py`

Some OVPhysX properties create `wp.array()` views into larger buffers (e.g., extracting position from a transform). These need special handling because the view is created each time from the parent data.

Check lines ~1400-1517 for properties like:
- `root_pos_w` (view into `root_link_pose_w`)
- `root_quat_w` (view into `root_link_pose_w`)
- `body_pos_w`, `body_quat_w` (views into `body_link_pose_w`)
- `body_lin_vel_w`, `body_ang_vel_w` (views into `body_link_vel_w`)

For these view properties, follow the same `_ta` cache pattern. The underlying view `wp.array` is stable across calls (same pointer), so the TorchArray cache is valid.

- [ ] **Step 1: Convert view-based properties**

- [ ] **Step 2: Commit**

```
git add source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py
git commit -m "Convert OVPhysX view-based properties to return TorchArray"
```

---

### Task 5: Update tests and add GRAVITY_VEC_W / FORWARD_VEC_B test

**Files:**
- Modify: `source/isaaclab_ovphysx/test/assets/test_articulation_data.py`

- [ ] **Step 1: Fix existing test**

The existing `test_joint_acc_uses_inverse_dt` calls `data.joint_acc.numpy()`. Since `joint_acc` now returns `TorchArray`, change to `data.joint_acc.warp.numpy()`.

- [ ] **Step 2: Add GRAVITY_VEC_W / FORWARD_VEC_B test**

```python
def test_gravity_and_forward_are_torch_array(self):
    """GRAVITY_VEC_W and FORWARD_VEC_B should be TorchArray instances."""
    from isaaclab.utils.warp import TorchArray

    mock_bindings = MockOvPhysxBindingSet(num_instances=2, num_joints=1, num_bodies=1)
    data = ArticulationData(mock_bindings.bindings, device="cpu")
    data._create_buffers()

    assert isinstance(data.GRAVITY_VEC_W, TorchArray)
    assert isinstance(data.FORWARD_VEC_B, TorchArray)
    assert data.GRAVITY_VEC_W.torch.shape == (2, 3)
    assert data.FORWARD_VEC_B.torch.shape == (2, 3)
```

- [ ] **Step 3: Run tests**

Run: `docker exec isaac-lab-base bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/assets/test_articulation_data.py -v"`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```
git add source/isaaclab_ovphysx/test/assets/test_articulation_data.py
git commit -m "Update OVPhysX tests for TorchArray returns"
```

---

### Task 6: Update changelog and version

**Files:**
- Modify: `source/isaaclab_ovphysx/docs/CHANGELOG.rst`
- Modify: `source/isaaclab_ovphysx/config/extension.toml`

- [ ] **Step 1: Add changelog entry**

Add a new version entry above the existing 0.1.0 entry:

```rst
0.1.1 (2026-04-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* :class:`~isaaclab_ovphysx.assets.articulation.ArticulationData` properties now
  return :class:`~isaaclab.utils.warp.TorchArray` instead of raw ``wp.array``.
```

- [ ] **Step 2: Bump extension.toml version**

Change `version = "0.1.0"` to `version = "0.1.1"` in `source/isaaclab_ovphysx/config/extension.toml`.

- [ ] **Step 3: Run pre-commit**

Run: `./isaaclab.sh -f`
Expected: All checks pass.

- [ ] **Step 4: Commit**

```
git add source/isaaclab_ovphysx/docs/CHANGELOG.rst source/isaaclab_ovphysx/config/extension.toml
git commit -m "Add OVPhysX TorchArray changelog entry and bump to 0.1.1"
```

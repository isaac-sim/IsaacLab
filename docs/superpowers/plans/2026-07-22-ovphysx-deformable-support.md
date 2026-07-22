# OVPhysX Deformable Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PhysX-style volume and surface deformable-object support to the OVPhysX backend using dtype-aware OVPhysX 0.5 tensor bindings.

**Architecture:** Make `OvPhysxView` derive scalar layout from each binding's DLPack metadata, then compose it behind deformable-specific body and material adapters. Implement the backend asset and lazy data container against those adapters, and switch the manager to full-stage loading only for scenes containing deformables.

**Tech Stack:** Python 3.12, Warp 1.15, PyTorch 2.11, OpenUSD 25.11, OVPhysX 0.5 tensor bindings, pytest, Ruff, pre-commit.

## Global Constraints

- Base all work on current `origin/develop`; use Marco's branch as a reference, not as a commit to cherry-pick.
- Do not edit `pyproject.toml`, `uv.lock`, or dependency-resolution workflows; PR #6660 owns the OVPhysX/Warp upgrade.
- Use `/tmp/isaaclab-ovphysx-deformable-support/.venv` with `/home/antoiner/ovphysx-0.5.2+head.f62c22207c-py3-none-manylinux_2_35_x86_64.whl` for development.
- Keep `isaaclab_ovphysx` kitless: no `omni.physics.tensors`, Kit, or AppLauncher imports.
- Support pre-authored volume and surface deformables; do not add procedural tetrahedralization.
- Match PhysX public method names and error behavior where OVPhysX exposes equivalent capabilities.
- Preserve OVPhysX's native device policy and full-buffer indexed/masked write contract.
- Use wheel-reported dtype metadata; do not add a per-`TensorType` scalar dtype table.
- New Python files use the 2026 Isaac Lab SPDX header, PEP 8, modern type hints, Google-style docstrings, and SI units on public physical quantities.
- Follow red-green-refactor for every behavior and verify each regression test fails before its implementation is added.
- Add one `source/isaaclab_ovphysx/changelog.d/antoiner-ovphysx-deformable.minor.rst` fragment; do not edit generated changelogs or extension versions.

---

## File map

- Modify `source/isaaclab_ovphysx/isaaclab_ovphysx/sim/views/ovphysx_view.py`: binding-reported scalar dtype allocation and validation.
- Modify `source/isaaclab_ovphysx/test/sim/test_ovphysx_view.py`: float32/int32/uint8 metadata regressions.
- Modify `source/isaaclab_ovphysx/isaaclab_ovphysx/tensor_types.py`: deformable aliases and CPU-resident material classification.
- Create `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/views.py`: PhysX-style body/material adapters over `OvPhysxView`.
- Create `source/isaaclab_ovphysx/test/assets/test_deformable_views.py`: isolated adapter tests with fake bindings.
- Modify `source/isaaclab_ovphysx/isaaclab_ovphysx/physics/ovphysx_manager.py`: neutral full-stage requirement and clone suppression.
- Modify `source/isaaclab_ovphysx/test/physics/test_ovphysx_scene_data_backend.py`: manager full-stage regressions.
- Create `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/kernels.py`: nodal copy/state/mean/target kernels.
- Create `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/deformable_object_data.py`: lazy nodal and derived state.
- Create `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/deformable_object.py`: OVPhysX backend asset.
- Create `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/__init__.py` and `__init__.pyi`: lazy module and public exports.
- Modify `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/__init__.pyi`: top-level public exports.
- Create `source/isaaclab_ovphysx/test/assets/test_deformable_object_helpers.py`: schema discovery, type detection, buffer, and error unit tests.
- Create `source/isaaclab_ovphysx/test/deformable_utils.py`: pre-authored volume/surface USD fixtures.
- Create `source/isaaclab_ovphysx/test/assets/test_deformable_object.py`: real CUDA volume/surface and replication tests.
- Create `source/isaaclab_ovphysx/test/tasks/__init__.py` and `test_lift_franka_soft_deformable.py`: task-level smoke.
- Create `source/isaaclab_ovphysx/changelog.d/antoiner-ovphysx-deformable.minor.rst`: user-facing entry.
- Modify `docs/source/api/lab_ovphysx/isaaclab_ovphysx.assets.rst`: public asset/data API entries.

---

### Task 1: Make `OvPhysxView` dtype-aware

**Files:**
- Modify: `source/isaaclab_ovphysx/test/sim/test_ovphysx_view.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/sim/views/ovphysx_view.py`

**Interfaces:**
- Consumes: `TensorBinding.dtype` with DLPack `code`, `bits`, and `lanes`; `TensorBinding.shape`.
- Produces: `_binding_scalar_dtype(binding: _BindingLike) -> Any`; dtype-aware `get_attribute`, `read_into`, and `set_attribute` without changing their public signatures.

- [ ] **Step 1: Extend the fake binding with dtype metadata**

Add a tiny immutable fake dtype and per-binding dtype selection:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class _FakeDtype:
    code: int
    bits: int
    lanes: int = 1


_FLOAT32 = _FakeDtype(code=2, bits=32)
_INT32 = _FakeDtype(code=0, bits=32)
_UINT8 = _FakeDtype(code=1, bits=8)

_DTYPES = {
    TensorType.DEFORMABLE_SIM_ELEMENT_INDICES: _INT32,
    TensorType.RIGID_BODY_DISABLE_SIMULATION: _UINT8,
}
```

Set `self.dtype = _DTYPES.get(tensor_type, _FLOAT32)` in `_FakeBinding.__init__`, and extend `_SHAPES` for the two non-float types.

- [ ] **Step 2: Add failing allocation and validation tests**

Add these behaviors:

```python
def test_get_attribute_uses_binding_reported_int32_dtype():
    view = _make_view(n=2)
    values = view.get_attribute(TensorType.DEFORMABLE_SIM_ELEMENT_INDICES)
    assert values.dtype == wp.int32
    assert tuple(values.shape) == (2, 1)


def test_get_attribute_uses_binding_reported_uint8_dtype():
    view = _make_view(n=2)
    values = view.get_attribute(TensorType.RIGID_BODY_DISABLE_SIMULATION)
    assert values.dtype == wp.uint8


def test_int32_binding_accepts_int32_and_rejects_float32():
    view = _make_view(n=2)
    view.read_into(
        TensorType.DEFORMABLE_SIM_ELEMENT_INDICES,
        wp.zeros((2, 1), dtype=wp.int32),
    )
    with pytest.raises(OvPhysxView.DtypeMismatch, match="int32"):
        view.read_into(
            TensorType.DEFORMABLE_SIM_ELEMENT_INDICES,
            wp.zeros((2, 1), dtype=wp.float32),
        )


def test_missing_or_unsupported_binding_dtype_raises_compatibility_error():
    view = _make_view(n=1)
    binding = view.binding_for(TensorType.RIGID_BODY_MASS)
    binding.dtype = _FakeDtype(code=2, bits=64)
    with pytest.raises(OvPhysxView.DtypeMismatch, match="DLPack"):
        view.get_attribute(TensorType.RIGID_BODY_MASS)
```

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest \
  source/isaaclab_ovphysx/test/sim/test_ovphysx_view.py \
  -k 'reported_int32 or reported_uint8 or int32_binding or unsupported_binding_dtype' -q
```

Expected: FAIL because `OvPhysxView` allocates `wp.float32` and rejects `wp.int32`/`wp.uint8` independently of binding metadata.

- [ ] **Step 4: Implement generic DLPack-to-Warp scalar resolution**

Add a generic mapping and resolver near the existing dtype helpers:

```python
_DLPACK_TO_WARP_SCALAR: dict[tuple[int, int, int], Any] = {
    (2, 32, 1): wp.float32,
    (0, 32, 1): wp.int32,
    (1, 8, 1): wp.uint8,
}


def _binding_scalar_dtype(binding: _BindingLike) -> Any:
    dtype = getattr(binding, "dtype", None)
    if dtype is None:
        raise OvPhysxView.DtypeMismatch(
            "OVPhysX binding does not expose DLPack dtype metadata; install the OVPhysX 0.5 dependency."
        )
    key = (int(dtype.code), int(dtype.bits), int(dtype.lanes))
    try:
        return _DLPACK_TO_WARP_SCALAR[key]
    except KeyError:
        raise OvPhysxView.DtypeMismatch(
            f"Unsupported OVPhysX DLPack dtype code={key[0]}, bits={key[1]}, lanes={key[2]}."
        ) from None
```

Update `_as_binding_view` to compare `getattr(arr.dtype, "_wp_scalar_type_", arr.dtype)` with `_binding_scalar_dtype(binding)`, compute byte counts using `wp.types.type_size_in_bytes`, and reinterpret using the reported scalar dtype. Update `_attribute_dtype` so structured `_ATTR_DTYPE` mappings are used only when their scalar type matches the binding scalar; otherwise allocate `binding.shape` with the binding scalar.

Update module/docstring wording from “float32 only” to “binding-reported DLPack scalar dtype,” retaining the no-conversion policy.

- [ ] **Step 5: Run the complete view suite and verify GREEN**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/sim/test_ovphysx_view.py -q
```

Expected: all existing float tests plus the new int32/uint8 tests PASS.

- [ ] **Step 6: Commit the dtype layer**

```bash
git add \
  source/isaaclab_ovphysx/isaaclab_ovphysx/sim/views/ovphysx_view.py \
  source/isaaclab_ovphysx/test/sim/test_ovphysx_view.py
git commit -m "Support OVPhysX binding dtypes"
```

---

### Task 2: Add deformable tensor aliases and adapters

**Files:**
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/tensor_types.py`
- Create: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/views.py`
- Create: `source/isaaclab_ovphysx/test/assets/test_deformable_views.py`

**Interfaces:**
- Consumes: dtype-aware `OvPhysxView`; OVPhysX `TensorType` members for volume, surface, and material tensors.
- Produces: `OvPhysxDeformableBodyView(physx_instance: Any, pattern: str, device: str, deformable_type: Literal["volume", "surface"])`; `OvPhysxDeformableMaterialView(physx_instance: Any, pattern: str)`.

- [ ] **Step 1: Write fake-binding adapter tests**

Create a fake PhysX object whose bindings report the exact shapes and dtypes below:

```python
_SPECS = {
    TensorType.DEFORMABLE_SIM_NODAL_POSITION: ((2, 5, 3), wp.float32),
    TensorType.DEFORMABLE_SIM_NODAL_VELOCITY: ((2, 5, 3), wp.float32),
    TensorType.DEFORMABLE_SIM_KINEMATIC_TARGET: ((2, 5, 4), wp.float32),
    TensorType.DEFORMABLE_REST_NODAL_POSITION: ((2, 5, 3), wp.float32),
    TensorType.DEFORMABLE_SIM_ELEMENT_INDICES: ((2, 2, 4), wp.int32),
    TensorType.DEFORMABLE_COLLISION_ELEMENT_INDICES: ((2, 6, 4), wp.int32),
    TensorType.SURFACE_DEFORMABLE_SIM_POSITION: ((2, 4, 3), wp.float32),
    TensorType.SURFACE_DEFORMABLE_SIM_VELOCITY: ((2, 4, 3), wp.float32),
    TensorType.SURFACE_DEFORMABLE_REST_POSITION: ((2, 4, 3), wp.float32),
    TensorType.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES: ((2, 2, 3), wp.int32),
}
```

Test these public behaviors:

```python
def test_volume_view_exposes_physx_style_dimensions_and_connectivity():
    view = OvPhysxDeformableBodyView(_FakePhysX(), "/World/env_*/Soft", "cpu", "volume")
    assert view.count == 2
    assert view.max_simulation_nodes_per_body == 5
    assert view.max_simulation_elements_per_body == 2
    assert view.max_collision_elements_per_body == 6
    assert view.get_simulation_element_indices().dtype == wp.int32


def test_surface_view_maps_common_methods_and_rejects_targets():
    view = OvPhysxDeformableBodyView(_FakePhysX(), "/World/env_*/Cloth", "cpu", "surface")
    assert tuple(view.get_simulation_nodal_positions().shape) == (2, 4, 3)
    assert tuple(view.get_simulation_element_indices().shape) == (2, 2, 3)
    with pytest.raises(ValueError, match="volume deformable"):
        view.get_simulation_nodal_kinematic_targets()


def test_body_view_forwards_full_buffer_and_indices():
    view = OvPhysxDeformableBodyView(_FakePhysX(), "/World/env_*/Soft", "cpu", "volume")
    values = wp.zeros((2, 5, 3), dtype=wp.float32)
    indices = wp.array([1], dtype=wp.int32)
    view.set_simulation_nodal_positions(values, indices=indices)
    assert view._view.binding_for(TensorType.DEFORMABLE_SIM_NODAL_POSITION).last_indices is indices


@pytest.mark.parametrize(
    "getter,setter",
    [
        ("get_dynamic_frictions", "set_dynamic_frictions"),
        ("get_youngs_moduli", "set_youngs_moduli"),
        ("get_poissons_ratios", "set_poissons_ratios"),
        ("get_elasticity_dampings", "set_elasticity_dampings"),
        ("get_bending_stiffnesses", "set_bending_stiffnesses"),
        ("get_thicknesses", "set_thicknesses"),
        ("get_bending_dampings", "set_bending_dampings"),
    ],
)
def test_material_view_exposes_all_properties_on_cpu(getter, setter):
    view = OvPhysxDeformableMaterialView(_FakePhysX(), "/World/env_*/material")
    values = getattr(view, getter)()
    assert str(values.device) == "cpu"
    getattr(view, setter)(wp.zeros((2,), dtype=wp.float32))
```

Also test Torch input conversion, CPU material staging, mask forwarding, and `destroy()` clearing every cached binding.

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/assets/test_deformable_views.py -q
```

Expected: collection ERROR because the deformable adapter module does not exist.

- [ ] **Step 3: Add tensor aliases and classifications**

Export direct aliases for every volume, surface, and material `TensorType`. Add the seven material tensor types to `_CPU_ONLY_TYPES_CANDIDATES`. Add rest-position and element-index names to `_READ_ONLY_NAMES` in `ovphysx_view.py`; do not mark nodal state or targets read-only.

The alias block must include:

```python
DEFORMABLE_SIM_NODAL_POSITION = _TT.DEFORMABLE_SIM_NODAL_POSITION
DEFORMABLE_SIM_NODAL_VELOCITY = _TT.DEFORMABLE_SIM_NODAL_VELOCITY
DEFORMABLE_SIM_KINEMATIC_TARGET = _TT.DEFORMABLE_SIM_KINEMATIC_TARGET
DEFORMABLE_REST_NODAL_POSITION = _TT.DEFORMABLE_REST_NODAL_POSITION
DEFORMABLE_SIM_ELEMENT_INDICES = _TT.DEFORMABLE_SIM_ELEMENT_INDICES
DEFORMABLE_COLLISION_ELEMENT_INDICES = _TT.DEFORMABLE_COLLISION_ELEMENT_INDICES
SURFACE_DEFORMABLE_SIM_POSITION = _TT.SURFACE_DEFORMABLE_SIM_POSITION
SURFACE_DEFORMABLE_SIM_VELOCITY = _TT.SURFACE_DEFORMABLE_SIM_VELOCITY
SURFACE_DEFORMABLE_REST_POSITION = _TT.SURFACE_DEFORMABLE_REST_POSITION
SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES = _TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES
DEFORMABLE_MATERIAL_DYNAMIC_FRICTION = _TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION
DEFORMABLE_MATERIAL_YOUNGS_MODULUS = _TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS
DEFORMABLE_MATERIAL_POISSONS_RATIO = _TT.DEFORMABLE_MATERIAL_POISSONS_RATIO
DEFORMABLE_MATERIAL_ELASTICITY_DAMPING = _TT.DEFORMABLE_MATERIAL_ELASTICITY_DAMPING
DEFORMABLE_MATERIAL_BENDING_STIFFNESS = _TT.DEFORMABLE_MATERIAL_BENDING_STIFFNESS
DEFORMABLE_MATERIAL_THICKNESS = _TT.DEFORMABLE_MATERIAL_THICKNESS
DEFORMABLE_MATERIAL_BENDING_DAMPING = _TT.DEFORMABLE_MATERIAL_BENDING_DAMPING
```

- [ ] **Step 4: Implement the body adapter over `OvPhysxView`**

Define immutable tensor maps:

```python
_BODY_TENSORS = {
    "volume": {
        "position": TT.DEFORMABLE_SIM_NODAL_POSITION,
        "velocity": TT.DEFORMABLE_SIM_NODAL_VELOCITY,
        "target": TT.DEFORMABLE_SIM_KINEMATIC_TARGET,
        "rest": TT.DEFORMABLE_REST_NODAL_POSITION,
        "elements": TT.DEFORMABLE_SIM_ELEMENT_INDICES,
        "collision_elements": TT.DEFORMABLE_COLLISION_ELEMENT_INDICES,
    },
    "surface": {
        "position": TT.SURFACE_DEFORMABLE_SIM_POSITION,
        "velocity": TT.SURFACE_DEFORMABLE_SIM_VELOCITY,
        "rest": TT.SURFACE_DEFORMABLE_REST_POSITION,
        "elements": TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES,
    },
}
```

Construct the shared view with:

```python
self._view = OvPhysxView(
    physx_instance,
    pattern=pattern,
    device=device,
    tensor_types=list(self._tensor_map.values()),
    eager=True,
)
```
Implement PhysX-style getters/setters by delegating to `get_attribute`, `read_into`, and `set_attribute`. Keep the legacy short aliases (`get_sim_nodal_positions`, `set_sim_nodal_positions`, and equivalents). Derive dimensions from binding shapes. Report unavailable surface collision dimensions as zero. Both get/set kinematic target methods must call a shared guard that raises `ValueError("Kinematic targets can only be set for volume deformable bodies.")` for surface assets.

Normalize Torch body positions and velocities with `wp.from_torch(values.contiguous(), dtype=wp.float32)` and kinematic targets with the same scalar dtype; move indices to the simulation device as `wp.int32`. Do not stage state between CPU and GPU.

- [ ] **Step 5: Implement the material adapter**

Construct a CPU `OvPhysxView` eagerly with all seven material tensors. Generate the repetitive getter/setter pairs from a private method but expose explicit typed public methods. Torch inputs are converted to contiguous float32 and explicitly copied to CPU before delegation. Resolve material indices as CPU `wp.int32` and masks as CPU `wp.bool`.

- [ ] **Step 6: Run view and adapter suites and verify GREEN**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest \
  source/isaaclab_ovphysx/test/sim/test_ovphysx_view.py \
  source/isaaclab_ovphysx/test/assets/test_deformable_views.py -q
```

Expected: both suites PASS with no dtype, device, or cleanup warnings.

- [ ] **Step 7: Commit adapters**

```bash
git add \
  source/isaaclab_ovphysx/isaaclab_ovphysx/tensor_types.py \
  source/isaaclab_ovphysx/isaaclab_ovphysx/sim/views/ovphysx_view.py \
  source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/views.py \
  source/isaaclab_ovphysx/test/assets/test_deformable_views.py
git commit -m "Add OVPhysX deformable views"
```

---

### Task 3: Add the full-stage deformable fallback

**Files:**
- Modify: `source/isaaclab_ovphysx/test/physics/test_ovphysx_scene_data_backend.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/physics/ovphysx_manager.py`

**Interfaces:**
- Consumes: authored live USD stage and pending runtime clone tuples.
- Produces: `OvPhysxManager.require_full_stage() -> None`; `_requires_full_stage: ClassVar[bool]`.

- [ ] **Step 1: Write manager regression tests**

Add tests that build an in-memory stage with `/World/envs/env_0/Cube` and `/World/envs/env_1/Cube` and prove:

```python
def test_manager_full_stage_requirement_preserves_authored_environments(tmp_path):
    stage = _make_two_environment_stage()
    output = tmp_path / "scene.usda"
    previous = OvPhysxManager._requires_full_stage
    try:
        OvPhysxManager._requires_full_stage = True
        OvPhysxManager._export_selected_stage(stage, str(output))
        exported = Usd.Stage.Open(str(output))
        assert exported.GetPrimAtPath("/World/envs/env_0/Cube").IsValid()
        assert exported.GetPrimAtPath("/World/envs/env_1/Cube").IsValid()
    finally:
        OvPhysxManager._requires_full_stage = previous


def test_manager_full_stage_requirement_discards_pending_runtime_clones():
    fake = SimpleNamespace(clone=lambda *args, **kwargs: pytest.fail("clone must not run"))
    previous = OvPhysxManager._pending_clones
    try:
        OvPhysxManager._pending_clones = [("/env_0", ["/env_1"], [(1.0, 0.0, 0.0)])]
        OvPhysxManager._replay_pending_clones(fake, requires_full_stage=True)
        assert OvPhysxManager._pending_clones == []
    finally:
        OvPhysxManager._pending_clones = previous


def test_manager_resets_full_stage_requirement_between_contexts():
    OvPhysxManager._requires_full_stage = True
    OvPhysxManager.close()
    assert OvPhysxManager._requires_full_stage is False
```

Preserve the existing env-0-only test as the negative case.

- [ ] **Step 2: Run manager tests and verify RED**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest \
  source/isaaclab_ovphysx/test/physics/test_ovphysx_scene_data_backend.py \
  -k 'full_stage or env0_only' -q
```

Expected: FAIL because the manager has no neutral full-stage flag or selection helpers.

- [ ] **Step 3: Implement the neutral manager flag and helpers**

Add:

```python
_requires_full_stage: ClassVar[bool] = False


@classmethod
def require_full_stage(cls) -> None:
    """Load every authored environment during the next stage warmup."""
    cls._requires_full_stage = True


@classmethod
def _export_selected_stage(cls, sim_stage: Any, target_file: str) -> None:
    if cls._requires_full_stage:
        sim_stage.Export(target_file)
    else:
        cls._export_env0_only_stage(sim_stage, target_file)


@classmethod
def _replay_pending_clones(cls, physx: Any, requires_full_stage: bool) -> None:
    if requires_full_stage:
        cls._pending_clones.clear()
        return
    for source, targets, parent_positions in cls._pending_clones:
        op_idx = physx.clone(source, targets, parent_positions=parent_positions)
        physx.wait_op(op_idx)
    cls._pending_clones.clear()
```

Use `_export_selected_stage` and `_replay_pending_clones` in `_warmup_and_load`. Reset the flag in `initialize` and `close`.

When rebasing after PR #6660, retain the public `require_full_stage` name and apply the same flag in the new serializer: return `sim_stage.Flatten().ExportToString()` unchanged for full stage, or delete envs 1..N for the normal path. Do not restore file-export terminology.

- [ ] **Step 4: Run the manager suite and verify GREEN**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/physics/test_ovphysx_scene_data_backend.py -q
```

Expected: all manager and scene-data tests PASS.

- [ ] **Step 5: Commit the manager fallback**

```bash
git add \
  source/isaaclab_ovphysx/isaaclab_ovphysx/physics/ovphysx_manager.py \
  source/isaaclab_ovphysx/test/physics/test_ovphysx_scene_data_backend.py
git commit -m "Load full stages for OVPhysX deformables"
```

---

### Task 4: Implement the OVPhysX deformable asset and data container

**Files:**
- Create: `source/isaaclab_ovphysx/test/assets/test_deformable_object_helpers.py`
- Create: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/kernels.py`
- Create: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/deformable_object_data.py`
- Create: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/deformable_object.py`
- Create: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/__init__.py`
- Create: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object/__init__.pyi`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/__init__.pyi`

**Interfaces:**
- Consumes: `BaseDeformableObject`, `BaseDeformableObjectData`, deformable adapters, `OvPhysxManager.require_full_stage`.
- Produces: `isaaclab_ovphysx.assets.deformable_object.DeformableObject`; `DeformableObjectData`; factory-resolvable `isaaclab.assets.DeformableObject` on OVPhysX.

- [ ] **Step 1: Write schema, type, surface-target, and buffer tests**

Create tests for `_get_api_schemas`, `_detect_deformable_type`, `_to_ovphysx_pattern`, and a shell object wired to fake views. Include:

```python
def test_get_api_schemas_includes_unregistered_authored_metadata():
    prim = _make_prim_with_api_metadata(["OmniPhysicsDeformableBodyAPI"])
    assert "OmniPhysicsDeformableBodyAPI" in _get_api_schemas(prim)


@pytest.mark.parametrize(
    "material_schemas,child_type,expected",
    [
        (["PhysxDeformableMaterialAPI"], "TetMesh", "volume"),
        (["PhysxSurfaceDeformableMaterialAPI"], "Mesh", "surface"),
        ([], "TetMesh", "volume"),
        ([], "Mesh", "surface"),
    ],
)
def test_detect_deformable_type_matches_physx_rules(material_schemas, child_type, expected):
    root, material = _make_deformable_prims(material_schemas, child_type)
    assert _detect_deformable_type(root, material) == expected


def test_surface_kinematic_target_write_matches_physx_error():
    asset = _make_asset_shell(deformable_type="surface")
    with pytest.raises(ValueError, match="Kinematic targets can only be set for volume deformable bodies"):
        asset.write_nodal_kinematic_target_to_sim_index(
            torch.zeros((2, 4, 4), device=asset.device)
        )


def test_indexed_position_write_updates_full_internal_buffer():
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=4)
    selected = torch.ones((1, 4, 3), device=asset.device)
    asset.write_nodal_pos_to_sim_index(selected, env_ids=torch.tensor([2], device=asset.device))
    assert torch.count_nonzero(asset.data.nodal_pos_w.torch[0:2]) == 0
    torch.testing.assert_close(asset.data.nodal_pos_w.torch[2], selected[0])
    assert asset.root_view.last_indices.tolist() == [2]
```

Also cover optional material `None`, volume target initialization to flag `1.0`, lazy root mean updates, cleanup, and factory export presence.

- [ ] **Step 2: Run helper tests and verify RED**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/assets/test_deformable_object_helpers.py -q
```

Expected: collection ERROR because the OVPhysX deformable asset package does not exist.

- [ ] **Step 3: Add focused Warp kernels**

Define `vec6f` and these kernels with exact responsibilities:

```python
vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def write_nodal_vec3f_to_buffer(
    data: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    full_data: bool,
    out_data: wp.array2d(dtype=wp.vec3f),
):
    i, j = wp.tid()
    out_data[env_ids[i], j] = data[env_ids[i], j] if full_data else data[i, j]


@wp.kernel
def write_nodal_vec4f_to_buffer(
    data: wp.array2d(dtype=wp.vec4f),
    env_ids: wp.array(dtype=wp.int32),
    full_data: bool,
    out_data: wp.array2d(dtype=wp.vec4f),
):
    i, j = wp.tid()
    out_data[env_ids[i], j] = data[env_ids[i], j] if full_data else data[i, j]
```

Also add `compute_nodal_state_w`, `compute_mean_vec3f_over_vertices`, and `set_kinematic_flags_to_one` using the algorithms in the approved design. Do not add `wp.printf`.

- [ ] **Step 4: Implement the lazy data container**

Subclass `BaseDeformableObjectData`, weak-reference the body view, allocate stable `TimestampedBufferWarp` arrays for `wp.vec3f`, `vec6f`, and root means, and cache one `ProxyArray` per public property.

Use adapter in-place reads:

```python
if self._nodal_pos_w.timestamp < self._sim_timestamp:
    self._root_view.read_simulation_nodal_positions_into(
        self._nodal_pos_w.data.view(wp.float32).reshape(
            (self._num_instances, self._max_sim_vertices, 3)
        )
    )
    self._nodal_pos_w.timestamp = self._sim_timestamp
```

Implement velocity identically, combine state with `compute_nodal_state_w`, and compute root means with `compute_mean_vec3f_over_vertices`. Public docstrings must state `[m]` and `[m/s]`.

- [ ] **Step 5: Implement the asset lifecycle and PhysX-compatible API**

Subclass `BaseDeformableObject`, set `__backend_name__ = "ovphysx"`, queue USD replication, call `OvPhysxManager.require_full_stage()`, and register `vec6f` in `_DTYPE_TO_TORCH_TRAILING_DIMS`.

Implement these public properties and methods with the same signatures as `source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object.py`:

```python
data
num_instances
num_bodies
root_view
root_physx_view
material_physx_view
max_sim_elements_per_body
max_collision_elements_per_body
max_sim_vertices_per_body
max_collision_vertices_per_body
reset
write_data_to_sim
update
write_nodal_state_to_sim_index
write_nodal_pos_to_sim_index
write_nodal_velocity_to_sim_index
write_nodal_kinematic_target_to_sim_index
```

In each indexed write, validate through `AssetBase.assert_shape_and_dtype`, copy selected rows into the complete internal buffer with the appropriate kernel, update/invalidate timestamps, and delegate the complete buffer plus `env_ids` to the body adapter. Rely on inherited mask and deprecated wrapper methods unless an OVPhysX-specific override is required by a test.

Initialization must:

1. Verify all required volume, surface, and material tensor enum members exist.
2. Obtain the live `OvPhysxManager` handle and CUDA device.
3. Resolve exactly one authored deformable body below the template using `_get_api_schemas`.
4. Resolve the optional direct physics material.
5. Detect volume or surface using schema-first, topology-second rules.
6. Convert Isaac Lab regex expressions to OVPhysX glob patterns.
7. Construct body and optional material adapters.
8. Construct `DeformableObjectData` and default buffers.
9. For volume only, initialize all kinematic flags to `1.0` and write the target buffer to OVPhysX.
10. Initialize debug visualization.

Surface debug visualization displays the below-ground sentinel because no targets exist. Invalidation destroys both adapters before clearing references.

- [ ] **Step 6: Add lazy exports and stubs**

Use `lazy_export()` in the runtime `__init__.py`. Export `DeformableObject` and `DeformableObjectData` in the subpackage `.pyi` and top-level `assets/__init__.pyi`. Do not expose adapter helper classes as top-level public API.

- [ ] **Step 7: Run helper, view, and manager suites and verify GREEN**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest \
  source/isaaclab_ovphysx/test/assets/test_deformable_object_helpers.py \
  source/isaaclab_ovphysx/test/assets/test_deformable_views.py \
  source/isaaclab_ovphysx/test/sim/test_ovphysx_view.py \
  source/isaaclab_ovphysx/test/physics/test_ovphysx_scene_data_backend.py -q
```

Expected: all tests PASS.

- [ ] **Step 8: Commit asset and data support**

```bash
git add \
  source/isaaclab_ovphysx/isaaclab_ovphysx/assets/__init__.pyi \
  source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object \
  source/isaaclab_ovphysx/test/assets/test_deformable_object_helpers.py
git commit -m "Add OVPhysX deformable assets"
```

---

### Task 5: Validate real volume, surface, replication, and task behavior

**Files:**
- Create: `source/isaaclab_ovphysx/test/deformable_utils.py`
- Create: `source/isaaclab_ovphysx/test/assets/test_deformable_object.py`
- Create: `source/isaaclab_ovphysx/test/tasks/__init__.py`
- Create: `source/isaaclab_ovphysx/test/tasks/test_lift_franka_soft_deformable.py`

**Interfaces:**
- Consumes: public `isaaclab.assets.DeformableObject`, authored USD schemas, real CUDA OVPhysX runtime.
- Produces: end-to-end regression coverage for both deformable types and cloned scenes.

- [ ] **Step 1: Add pre-authored volume and surface spawners**

Port the five-node/two-tetrahedron fixture from Marco's `deformable_utils.py`. Add a four-node/two-triangle surface fixture using `UsdGeom.Mesh` with:

```python
points = [
    Gf.Vec3f(0.0, 0.0, 0.0),
    Gf.Vec3f(0.2, 0.0, 0.0),
    Gf.Vec3f(0.2, 0.2, 0.0),
    Gf.Vec3f(0.0, 0.2, 0.0),
]
triangles = [Gf.Vec3i(0, 1, 2), Gf.Vec3i(0, 2, 3)]
mesh.CreatePointsAttr(points)
mesh.CreateFaceVertexCountsAttr([3, 3])
mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
```

Author `OmniPhysicsDeformableBodyAPI`, `OmniPhysicsSurfaceDeformableSimAPI`, `OmniPhysicsDeformablePoseAPI:default`, `PhysicsCollisionAPI`, and `MaterialBindingAPI`. Set `omniphysics:restShapePoints`, `omniphysics:restTriVtxIndices`, pose points/purposes, and velocities. Bind `PhysxSurfaceDeformableBodyMaterialCfg` with non-default values for every surface property.

- [ ] **Step 2: Write real-backend tests before running them**

Create CUDA-gated tests named `test_volume_deformable_reads_writes_targets_materials_and_steps`, `test_surface_deformable_reads_writes_materials_and_steps`, `test_deformable_interactive_scene_uses_full_authored_stage`, and `test_mixed_deformable_rigid_scene_does_not_duplicate_runtime_clones`.

The volume test must assert `int32` tetrahedral and collision connectivity, indexed position/velocity writes, initialized and indexed kinematic targets, all applicable material properties, finite state after five steps, and correct derived root shapes.

The surface test must assert `int32` triangular connectivity, `nodal_kinematic_target is None`, the PhysX-compatible target `ValueError`, all seven material properties, indexed position/velocity writes, and finite state after five steps.

The scene tests use three environments and assert both body/material counts are three. The mixed scene additionally asserts the rigid object count remains three.

- [ ] **Step 3: Run real tests and verify RED for any remaining integration gaps**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest \
  source/isaaclab_ovphysx/test/assets/test_deformable_object.py -vv
```

Expected before final integration fixes: at least one FAIL in binding discovery, authored surface schema ingestion, material availability, or full-stage replication. Confirm each failure is caused by the missing integration behavior rather than fixture syntax.

- [ ] **Step 4: Make minimal integration corrections**

Correct only behavior exposed by the failing real tests: schema metadata parsing, glob conversion, binding selection, material CPU staging, stage selection, or cleanup ordering. Do not weaken assertions or add backend-unrelated workarounds.

- [ ] **Step 5: Add and run the soft-lift task smoke**

Create a one-environment OVPhysX variant of `Isaac-Lift-Soft-Franka` using the authored volume spawner. Disable unrelated table/light/ground assets and arm IK action as Marco's smoke does. Assert finite reset observations, finite deformable and end-effector state, successful kinematic target readback, and three finite environment steps.

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest \
  source/isaaclab_ovphysx/test/tasks/test_lift_franka_soft_deformable.py -vv
```

Expected: PASS.

- [ ] **Step 6: Verify real suites GREEN**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest \
  source/isaaclab_ovphysx/test/assets/test_deformable_object.py \
  source/isaaclab_ovphysx/test/tasks/test_lift_franka_soft_deformable.py -vv
```

Expected: all CUDA-capable tests PASS; only explicit CUDA skips are acceptable when no GPU is present.

- [ ] **Step 7: Commit real-backend coverage**

```bash
git add \
  source/isaaclab_ovphysx/test/deformable_utils.py \
  source/isaaclab_ovphysx/test/assets/test_deformable_object.py \
  source/isaaclab_ovphysx/test/tasks
git commit -m "Test OVPhysX deformable workflows"
```

---

### Task 6: Add changelog, documentation, and final verification

**Files:**
- Create: `source/isaaclab_ovphysx/changelog.d/antoiner-ovphysx-deformable.minor.rst`
- Modify: `docs/source/api/lab_ovphysx/isaaclab_ovphysx.assets.rst`

**Interfaces:**
- Consumes: completed implementation and test suites.
- Produces: user-facing changelog and verified branch.

- [ ] **Step 1: Add the changelog fragment**

```rst
Added
^^^^^

* Added volume and surface deformable-object support for the OVPhysX backend,
  including nodal state, volume kinematic targets, mesh connectivity, and
  runtime deformable material properties.
```

- [ ] **Step 2: Compile every touched Python module**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m compileall -q \
  source/isaaclab_ovphysx/isaaclab_ovphysx/assets/deformable_object \
  source/isaaclab_ovphysx/isaaclab_ovphysx/sim/views/ovphysx_view.py \
  source/isaaclab_ovphysx/isaaclab_ovphysx/physics/ovphysx_manager.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Run the focused regression suite**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest \
  source/isaaclab_ovphysx/test/sim/test_ovphysx_view.py \
  source/isaaclab_ovphysx/test/physics/test_ovphysx_scene_data_backend.py \
  source/isaaclab_ovphysx/test/assets/test_deformable_views.py \
  source/isaaclab_ovphysx/test/assets/test_deformable_object_helpers.py \
  source/isaaclab_ovphysx/test/assets/test_deformable_object.py \
  source/isaaclab_ovphysx/test/tasks/test_lift_franka_soft_deformable.py -vv
```

Expected: all unit and available CUDA tests PASS.

- [ ] **Step 4: Document and regenerate the public asset API**

Expand `docs/source/api/lab_ovphysx/isaaclab_ovphysx.assets.rst` to mirror the existing PhysX asset page for `DeformableObject` and `DeformableObjectData`, including `:members:`, `:inherited-members:`, and `:show-inheritance:`. State that `isaaclab.assets.DeformableObjectCfg` is shared while schema/material spawn configuration remains backend-specific.

Run:

```bash
./isaaclab.sh -d
```

Review the generated documentation diff and retain only changes produced for the new OVPhysX deformable exports.

- [ ] **Step 5: Run mandatory pre-commit twice when necessary**

Run:

```bash
source .venv/bin/activate
./isaaclab.sh -f
git status --short
```

If hooks modify files, review the diff, stage those files, and run `./isaaclab.sh -f` again. Expected final result: every hook PASS and no unstaged formatter changes.

- [ ] **Step 6: Commit the changelog and verification-driven fixes**

```bash
git add source/isaaclab_ovphysx/changelog.d/antoiner-ovphysx-deformable.minor.rst
git add docs/source/api/lab_ovphysx/isaaclab_ovphysx.assets.rst
git commit -m "Document OVPhysX deformable support"
```

- [ ] **Step 7: Review the complete branch against the specification**

Run:

```bash
git diff --check origin/develop...HEAD
git diff --stat origin/develop...HEAD
git status --short
```

Confirm no dependency pins, lock files, generated changelogs, or unrelated packages changed. Record test counts and any hardware skips in the final handoff.

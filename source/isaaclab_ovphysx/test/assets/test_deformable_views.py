# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVPhysX deformable body and material tensor adapters."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_ovphysx.assets.deformable_object.views import (  # noqa: E402
    OvPhysxDeformableBodyView,
    OvPhysxDeformableMaterialView,
)
from ovphysx.types import TensorType  # noqa: E402

wp.init()
wp.set_device("cpu")


@dataclass(frozen=True)
class _FakeDtype:
    code: int
    bits: int
    lanes: int = 1


_FLOAT32 = _FakeDtype(code=2, bits=32)
_INT32 = _FakeDtype(code=0, bits=32)

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
    TensorType.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION: ((2,), wp.float32),
    TensorType.DEFORMABLE_MATERIAL_YOUNGS_MODULUS: ((2,), wp.float32),
    TensorType.DEFORMABLE_MATERIAL_POISSONS_RATIO: ((2,), wp.float32),
    TensorType.DEFORMABLE_MATERIAL_ELASTICITY_DAMPING: ((2,), wp.float32),
    TensorType.DEFORMABLE_MATERIAL_BENDING_STIFFNESS: ((2,), wp.float32),
    TensorType.DEFORMABLE_MATERIAL_THICKNESS: ((2,), wp.float32),
    TensorType.DEFORMABLE_MATERIAL_BENDING_DAMPING: ((2,), wp.float32),
}


class _FakeBinding:
    """Minimal tensor binding that records reads, writes, and cleanup."""

    def __init__(self, tensor_type: TensorType):
        self.tensor_type = tensor_type
        self.shape, dtype = _SPECS[tensor_type]
        self.dtype = _FLOAT32 if dtype == wp.float32 else _INT32
        self.count = self.shape[0]
        self.prim_paths = [f"/World/env_{index}/Soft" for index in range(self.count)]
        self.dof_names: list[str] = []
        self.body_names: list[str] = []
        self.joint_names: list[str] = []
        self.dof_count = 0
        self.body_count = 0
        self.joint_count = 0
        self.is_fixed_base = False
        self.fixed_tendon_count = 0
        self.spatial_tendon_count = 0
        self.last_indices: wp.array | None = None
        self.last_mask: wp.array | None = None
        self.last_values: wp.array | None = None
        self.destroyed = False

    def read(self, values: wp.array) -> None:
        pass

    def write(self, values: wp.array, indices: wp.array | None = None, mask: wp.array | None = None) -> None:
        self.last_values = values
        self.last_indices = indices
        self.last_mask = mask

    def destroy(self) -> None:
        self.destroyed = True


class _FakePhysX:
    """Fake PhysX instance returning the requested binding type."""

    def __init__(self):
        self.bindings: dict[TensorType, _FakeBinding] = {}

    def create_tensor_binding(self, *, tensor_type: TensorType, pattern: str | None = None) -> _FakeBinding:
        binding = _FakeBinding(tensor_type)
        self.bindings[tensor_type] = binding
        return binding


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


def test_body_view_forwards_full_buffer_indices_and_masks():
    view = OvPhysxDeformableBodyView(_FakePhysX(), "/World/env_*/Soft", "cpu", "volume")
    values = wp.zeros((2, 5, 3), dtype=wp.float32)
    indices = wp.array([1], dtype=wp.int32)
    mask = wp.array([False, True], dtype=wp.bool)

    view.set_simulation_nodal_positions(values, indices=indices)
    binding = view._view.binding_for(TensorType.DEFORMABLE_SIM_NODAL_POSITION)
    assert binding.last_indices is indices
    view.set_simulation_nodal_positions(values, mask=mask)
    assert binding.last_mask is mask


def test_body_view_converts_torch_state_and_target_inputs():
    view = OvPhysxDeformableBodyView(_FakePhysX(), "/World/env_*/Soft", "cpu", "volume")
    positions = torch.zeros((2, 3, 5), dtype=torch.float32).transpose(1, 2)
    targets = torch.zeros((2, 4, 5), dtype=torch.float32).transpose(1, 2)
    assert not positions.is_contiguous()
    assert not targets.is_contiguous()

    view.set_simulation_nodal_positions(positions)
    view.set_simulation_nodal_kinematic_targets(targets, indices=torch.tensor([1], dtype=torch.int64))

    position_binding = view._view.binding_for(TensorType.DEFORMABLE_SIM_NODAL_POSITION)
    target_binding = view._view.binding_for(TensorType.DEFORMABLE_SIM_KINEMATIC_TARGET)
    assert position_binding.last_indices is None
    assert position_binding.last_values.is_contiguous
    assert position_binding.last_values.dtype == wp.float32
    assert target_binding.last_values.is_contiguous
    assert target_binding.last_values.dtype == wp.float32
    assert str(target_binding.last_indices.device) == "cpu"
    assert target_binding.last_indices.dtype == wp.int32


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
def test_material_view_exposes_all_properties_on_cpu(getter: str, setter: str):
    view = OvPhysxDeformableMaterialView(_FakePhysX(), "/World/env_*/material")

    values = getattr(view, getter)()
    assert str(values.device) == "cpu"
    getattr(view, setter)(wp.zeros((2,), dtype=wp.float32))


def test_material_view_converts_torch_inputs_and_cpu_selection():
    view = OvPhysxDeformableMaterialView(_FakePhysX(), "/World/env_*/material")
    values = torch.zeros((2, 2), dtype=torch.float32)[:, 0]
    assert not values.is_contiguous()
    indices = torch.tensor([1], dtype=torch.int64)
    mask = torch.tensor([False, True])

    view.set_dynamic_frictions(values, indices=indices, mask=mask)
    binding = view._view.binding_for(TensorType.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION)
    assert binding.last_values.is_contiguous
    assert binding.last_values.dtype == wp.float32
    assert str(binding.last_indices.device) == "cpu"
    assert binding.last_indices.dtype == wp.int32
    assert str(binding.last_mask.device) == "cpu"
    assert binding.last_mask.dtype == wp.bool


def test_destroy_clears_every_cached_binding():
    physx = _FakePhysX()
    body_view = OvPhysxDeformableBodyView(physx, "/World/env_*/Soft", "cpu", "volume")
    body_view.destroy()

    assert all(binding.destroyed for binding in physx.bindings.values())
    assert body_view._view is None

    physx = _FakePhysX()
    material_view = OvPhysxDeformableMaterialView(physx, "/World/env_*/material")
    material_view.destroy()

    assert all(binding.destroyed for binding in physx.bindings.values())
    assert material_view._view is None

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the kitless OVPhysX deformable object asset."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_ovphysx import tensor_types as TT  # noqa: E402
from isaaclab_ovphysx.assets.deformable_object.deformable_object import DeformableObject  # noqa: E402
from isaaclab_ovphysx.assets.deformable_object.deformable_object_data import (  # noqa: E402
    DeformableObjectData,
)
from isaaclab_ovphysx.assets.deformable_object.kernels import vec6f  # noqa: E402

from isaaclab.assets.deformable_object import (  # noqa: E402
    BaseDeformableObject,
    BaseDeformableObjectData,
)
from isaaclab.assets.deformable_object import (  # noqa: E402
    DeformableObject as DeformableObjectFactory,
)

wp.init()
wp.set_device("cpu")


class _FakeBodyView:
    """Minimal generic OVPhysX view used by the asset shell."""

    def __init__(self, num_instances: int, num_vertices: int) -> None:
        self.count = num_instances
        self.max_simulation_nodes_per_body = num_vertices
        self.max_simulation_elements_per_body = 2
        self.max_collision_elements_per_body = 3
        self.max_collision_nodes_per_body = 0
        self.positions = wp.zeros((num_instances, num_vertices, 3), dtype=wp.float32, device="cpu")
        self.velocities = wp.zeros((num_instances, num_vertices, 3), dtype=wp.float32, device="cpu")
        self.targets = wp.zeros((num_instances, num_vertices, 4), dtype=wp.float32, device="cpu")
        self.position_reads = 0
        self.velocity_reads = 0
        self.position_write_count = 0
        self.velocity_write_count = 0
        self.target_write_count = 0
        self.last_indices: torch.Tensor | None = None
        self.last_values: wp.array | None = None

    def binding_for(self, tensor_type):
        if tensor_type in (TT.DEFORMABLE_SIM_ELEMENT_INDICES, TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES):
            shape = (self.count, self.max_simulation_elements_per_body, 4)
        elif tensor_type == TT.DEFORMABLE_COLLISION_ELEMENT_INDICES:
            shape = (self.count, self.max_collision_elements_per_body, 4)
        elif tensor_type == TT.DEFORMABLE_SIM_KINEMATIC_TARGET:
            shape = (self.count, self.max_simulation_nodes_per_body, 4)
        else:
            shape = (self.count, self.max_simulation_nodes_per_body, 3)
        return SimpleNamespace(shape=shape, count=self.count)

    def read_into(self, tensor_type, values: wp.array) -> None:
        if tensor_type in (TT.DEFORMABLE_SIM_NODAL_POSITION, TT.SURFACE_DEFORMABLE_SIM_POSITION):
            self.position_reads += 1
            wp.copy(values, self.positions)
        elif tensor_type in (TT.DEFORMABLE_SIM_NODAL_VELOCITY, TT.SURFACE_DEFORMABLE_SIM_VELOCITY):
            self.velocity_reads += 1
            wp.copy(values, self.velocities)
        else:
            raise AssertionError(f"Unexpected tensor read: {tensor_type}")

    def get_attribute(self, tensor_type) -> wp.array:
        if tensor_type == TT.DEFORMABLE_SIM_KINEMATIC_TARGET:
            return self.targets
        if tensor_type in (TT.DEFORMABLE_SIM_NODAL_POSITION, TT.SURFACE_DEFORMABLE_SIM_POSITION):
            return self.positions
        if tensor_type in (TT.DEFORMABLE_SIM_NODAL_VELOCITY, TT.SURFACE_DEFORMABLE_SIM_VELOCITY):
            return self.velocities
        binding = self.binding_for(tensor_type)
        return wp.zeros(binding.shape, dtype=wp.int32, device="cpu")

    def set_attribute(
        self,
        tensor_type,
        values: wp.array,
        indices: wp.array(dtype=wp.int32) | None = None,
        mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        self.last_values = values
        self.last_indices = wp.to_torch(indices) if indices is not None else None
        if tensor_type in (TT.DEFORMABLE_SIM_NODAL_POSITION, TT.SURFACE_DEFORMABLE_SIM_POSITION):
            self.position_write_count += 1
        elif tensor_type in (TT.DEFORMABLE_SIM_NODAL_VELOCITY, TT.SURFACE_DEFORMABLE_SIM_VELOCITY):
            self.velocity_write_count += 1
        elif tensor_type == TT.DEFORMABLE_SIM_KINEMATIC_TARGET:
            self.target_write_count += 1
        else:
            raise AssertionError(f"Unexpected tensor write: {tensor_type}")


class _FakeMaterialView:
    """Minimal optional plain material view."""

    count = 1


class _FakeVisualizer:
    """Capture marker positions from the debug visualization callback."""

    def __init__(self) -> None:
        self.positions: torch.Tensor | None = None

    def visualize(self, positions: torch.Tensor) -> None:
        self.positions = positions


def _make_asset_shell(
    *,
    deformable_type: str,
    num_instances: int = 2,
    num_vertices: int = 4,
    material_view: _FakeMaterialView | None = None,
) -> DeformableObject:
    asset = object.__new__(DeformableObject)
    asset._device = "cpu"
    asset._check_shapes = True
    asset._DTYPE_TO_TORCH_TRAILING_DIMS = {**asset._DTYPE_TO_TORCH_TRAILING_DIMS, vec6f: (6,)}
    asset._deformable_type = deformable_type
    if deformable_type == "volume":
        asset._sim_nodal_position_type = TT.DEFORMABLE_SIM_NODAL_POSITION
        asset._sim_nodal_velocity_type = TT.DEFORMABLE_SIM_NODAL_VELOCITY
        asset._sim_kinematic_target_type = TT.DEFORMABLE_SIM_KINEMATIC_TARGET
    else:
        asset._sim_nodal_position_type = TT.SURFACE_DEFORMABLE_SIM_POSITION
        asset._sim_nodal_velocity_type = TT.SURFACE_DEFORMABLE_SIM_VELOCITY
        asset._sim_kinematic_target_type = None
    asset._root_physx_view = _FakeBodyView(num_instances, num_vertices)
    asset._material_physx_view = material_view
    asset._data = DeformableObjectData(
        asset._root_physx_view,
        asset._device,
        position_tensor_type=asset._sim_nodal_position_type,
        velocity_tensor_type=asset._sim_nodal_velocity_type,
    )
    asset._ALL_INDICES = wp.array(range(num_instances), dtype=wp.int32, device=asset.device)
    asset._nodal_pos_w_f32 = None
    asset._nodal_vel_w_f32 = None
    asset._is_initialized = True
    asset._debug_vis_handle = None
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None
    return asset


def test_unbound_material_is_optional():
    asset = _make_asset_shell(deformable_type="volume")

    assert asset.material_physx_view is None


def test_surface_kinematic_target_write_matches_physx_error():
    asset = _make_asset_shell(deformable_type="surface")
    with pytest.raises(ValueError, match="Kinematic targets can only be set for volume deformable bodies"):
        asset.write_nodal_kinematic_target_to_sim_index(torch.zeros((2, 4, 4), device=asset.device))


def test_indexed_position_write_updates_full_internal_buffer():
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=4)
    selected = torch.ones((1, 4, 3), device=asset.device)
    asset.write_nodal_pos_to_sim_index(selected, env_ids=torch.tensor([2], device=asset.device))
    assert torch.count_nonzero(asset.data.nodal_pos_w.torch[0:2]) == 0
    torch.testing.assert_close(asset.data.nodal_pos_w.torch[2], selected[0])
    assert asset.root_view.last_indices.tolist() == [2]


def test_indexed_state_write_updates_position_and_velocity_buffers():
    asset = _make_asset_shell(deformable_type="volume", num_instances=2, num_vertices=4)
    selected = torch.cat(
        (torch.full((1, 4, 3), 2.0), torch.full((1, 4, 3), -3.0)),
        dim=-1,
    )

    asset.write_nodal_state_to_sim_index(selected, env_ids=[1])

    torch.testing.assert_close(asset.data.nodal_pos_w.torch[1], selected[0, :, :3])
    torch.testing.assert_close(asset.data.nodal_vel_w.torch[1], selected[0, :, 3:])
    assert asset.data._nodal_state_w.timestamp == -1.0


def test_indexed_state_write_refreshes_unselected_stale_cache_rows():
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=2)
    initial_positions = torch.full((3, 2, 3), -1.0, device=asset.device)
    initial_velocities = torch.full((3, 2, 3), -2.0, device=asset.device)
    asset.root_view.positions = wp.from_torch(initial_positions.contiguous(), dtype=wp.float32)
    asset.root_view.velocities = wp.from_torch(initial_velocities.contiguous(), dtype=wp.float32)

    asset.data.nodal_pos_w
    asset.data.nodal_vel_w

    latest_positions = torch.tensor(
        [
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
            [[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]],
            [[30.0, 31.0, 32.0], [33.0, 34.0, 35.0]],
        ],
        device=asset.device,
    )
    latest_velocities = torch.tensor(
        [
            [[-10.0, -11.0, -12.0], [-13.0, -14.0, -15.0]],
            [[-20.0, -21.0, -22.0], [-23.0, -24.0, -25.0]],
            [[-30.0, -31.0, -32.0], [-33.0, -34.0, -35.0]],
        ],
        device=asset.device,
    )
    asset.root_view.positions = wp.from_torch(latest_positions.contiguous(), dtype=wp.float32)
    asset.root_view.velocities = wp.from_torch(latest_velocities.contiguous(), dtype=wp.float32)
    asset.update(0.1)

    selected_state = torch.cat(
        (
            torch.full((1, 2, 3), 100.0, device=asset.device),
            torch.full((1, 2, 3), -100.0, device=asset.device),
        ),
        dim=-1,
    )
    asset.write_nodal_state_to_sim_index(selected_state, env_ids=[0])

    expected_positions = latest_positions.clone()
    expected_positions[0] = selected_state[0, :, :3]
    expected_velocities = latest_velocities.clone()
    expected_velocities[0] = selected_state[0, :, 3:]

    assert asset.data._nodal_pos_w.timestamp == asset.data._sim_timestamp
    assert asset.data._nodal_vel_w.timestamp == asset.data._sim_timestamp
    torch.testing.assert_close(asset.data.nodal_pos_w.torch, expected_positions)
    torch.testing.assert_close(asset.data.nodal_vel_w.torch, expected_velocities)
    torch.testing.assert_close(
        asset.data.nodal_state_w.torch, torch.cat((expected_positions, expected_velocities), dim=-1)
    )
    torch.testing.assert_close(asset.data.root_pos_w.torch, expected_positions.mean(dim=1))
    torch.testing.assert_close(asset.data.root_vel_w.torch, expected_velocities.mean(dim=1))


@pytest.mark.parametrize(
    ("property_name", "write_method_name", "simulator_attribute", "command_value"),
    [
        ("nodal_pos_w", "write_nodal_pos_to_sim_index", "positions", 100.0),
        ("nodal_vel_w", "write_nodal_velocity_to_sim_index", "velocities", -100.0),
    ],
)
def test_indexed_full_data_write_preserves_retained_aliased_edits(
    property_name: str, write_method_name: str, simulator_attribute: str, command_value: float
) -> None:
    """A retained public buffer preserves selected edits while stale rows hydrate."""
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=2)
    initial = torch.full((3, 2, 3), -1.0, device=asset.device)
    setattr(asset.root_view, simulator_attribute, wp.from_torch(initial.contiguous(), dtype=wp.float32))
    retained = getattr(asset.data, property_name)
    retained_torch = retained.torch

    latest = torch.arange(18, dtype=torch.float32, device=asset.device).reshape(3, 2, 3)
    setattr(asset.root_view, simulator_attribute, wp.from_torch(latest.contiguous(), dtype=wp.float32))
    asset.update(0.1)
    retained_torch[1].fill_(command_value)

    getattr(asset, write_method_name)(retained, env_ids=[1], full_data=True)

    expected = latest.clone()
    expected[1].fill_(command_value)
    assert getattr(asset.data, property_name) is retained
    torch.testing.assert_close(retained.torch, expected)


@pytest.mark.parametrize(
    ("property_name", "write_method_name", "simulator_attribute", "command_value"),
    [
        ("nodal_pos_w", "write_nodal_pos_to_sim_index", "positions", 100.0),
        ("nodal_vel_w", "write_nodal_velocity_to_sim_index", "velocities", -100.0),
    ],
)
def test_indexed_partial_write_preserves_retained_aliased_slice(
    property_name: str, write_method_name: str, simulator_attribute: str, command_value: float
) -> None:
    """A retained selected slice survives hydration of stale unselected rows."""
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=2)
    initial = torch.full((3, 2, 3), -1.0, device=asset.device)
    setattr(asset.root_view, simulator_attribute, wp.from_torch(initial.contiguous(), dtype=wp.float32))
    retained = getattr(asset.data, property_name)

    latest = torch.arange(18, dtype=torch.float32, device=asset.device).reshape(3, 2, 3)
    setattr(asset.root_view, simulator_attribute, wp.from_torch(latest.contiguous(), dtype=wp.float32))
    asset.update(0.1)
    selected = retained.torch[1:2]
    selected.fill_(command_value)

    getattr(asset, write_method_name)(selected, env_ids=[1])

    expected = latest.clone()
    expected[1].fill_(command_value)
    assert getattr(asset.data, property_name) is retained
    torch.testing.assert_close(retained.torch, expected)


def test_full_overwrite_stale_cache_does_not_read_simulator() -> None:
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=2)

    asset.data.nodal_pos_w
    asset.data.nodal_vel_w
    position_reads = asset.root_view.position_reads
    velocity_reads = asset.root_view.velocity_reads
    asset.update(0.1)

    full_state = torch.arange(36, dtype=torch.float32, device=asset.device).reshape(3, 2, 6)
    asset.write_nodal_state_to_sim_index(full_state, full_data=True)

    assert asset.root_view.position_reads == position_reads
    assert asset.root_view.velocity_reads == velocity_reads
    torch.testing.assert_close(asset.data.nodal_state_w.torch, full_state)


@pytest.mark.parametrize("trailing_dimension", [4, 5, 7])
def test_malformed_state_write_fails_before_mutating_or_writing(trailing_dimension: int):
    asset = _make_asset_shell(deformable_type="volume", num_instances=2, num_vertices=4)
    original_positions = asset.data.nodal_pos_w.torch.clone()
    original_velocities = asset.data.nodal_vel_w.torch.clone()
    malformed_state = torch.ones((1, 4, trailing_dimension), device=asset.device)

    with pytest.raises(AssertionError, match="nodal_state.*Shape mismatch"):
        asset.write_nodal_state_to_sim_index(malformed_state, env_ids=[1])

    torch.testing.assert_close(asset.data.nodal_pos_w.torch, original_positions)
    torch.testing.assert_close(asset.data.nodal_vel_w.torch, original_velocities)
    assert asset.root_view.position_write_count == 0
    assert asset.root_view.velocity_write_count == 0


def test_volume_target_initialization_sets_free_flags_and_writes_full_buffer():
    asset = _make_asset_shell(deformable_type="volume")

    asset._create_buffers()

    assert asset.data.nodal_kinematic_target is not None
    torch.testing.assert_close(
        asset.data.nodal_kinematic_target.torch[..., 3],
        torch.ones((asset.num_instances, asset.max_sim_vertices_per_body)),
    )
    assert asset.root_view.target_write_count == 1
    assert asset.root_view.last_indices is None


def test_surface_buffer_initialization_has_no_kinematic_target():
    asset = _make_asset_shell(deformable_type="surface")

    asset._create_buffers()

    assert asset.data.nodal_kinematic_target is None
    assert asset.root_view.target_write_count == 0


def test_lazy_root_means_refresh_in_place_after_update():
    root_view = _FakeBodyView(num_instances=2, num_vertices=2)
    root_view.positions = wp.array(
        [[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]], [[1.0, 3.0, 5.0], [3.0, 5.0, 7.0]]],
        dtype=wp.float32,
    )
    data = DeformableObjectData(
        root_view,
        "cpu",
        position_tensor_type=TT.DEFORMABLE_SIM_NODAL_POSITION,
        velocity_tensor_type=TT.DEFORMABLE_SIM_NODAL_VELOCITY,
    )

    root_pos = data.root_pos_w
    torch.testing.assert_close(root_pos.torch, torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))
    assert root_view.position_reads == 1

    root_view.positions = wp.array(
        [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]], [[6.0, 6.0, 6.0], [8.0, 8.0, 8.0]]],
        dtype=wp.float32,
    )
    data.update(0.1)

    assert data.root_pos_w is root_pos
    torch.testing.assert_close(root_pos.torch, torch.tensor([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]]))
    assert root_view.position_reads == 2


def test_surface_debug_visualization_uses_below_ground_sentinel():
    asset = _make_asset_shell(deformable_type="surface")
    asset.target_visualizer = _FakeVisualizer()

    asset._debug_vis_callback(None)

    torch.testing.assert_close(asset.target_visualizer.positions, torch.tensor([[0.0, 0.0, -10.0]]))


def test_invalidation_releases_body_and_material_views(monkeypatch: pytest.MonkeyPatch):
    asset = _make_asset_shell(deformable_type="volume", material_view=_FakeMaterialView())
    monkeypatch.setattr(BaseDeformableObject, "_invalidate_initialize_callback", lambda self, event: None)

    asset._invalidate_initialize_callback(None)

    assert asset._root_physx_view is None
    assert asset._material_physx_view is None


def test_data_implements_backend_neutral_interface():
    asset = _make_asset_shell(deformable_type="volume")
    assert isinstance(asset.data, BaseDeformableObjectData)


def test_factory_export_is_present():
    backend_module = importlib.import_module("isaaclab_ovphysx.assets.deformable_object")
    assets_module = importlib.import_module("isaaclab_ovphysx.assets")

    assert backend_module.DeformableObject is DeformableObject
    assert assets_module.DeformableObject is DeformableObject
    assert DeformableObjectFactory._get_module_name("ovphysx") == "isaaclab_ovphysx.assets.deformable_object"
    assert DeformableObject.__backend_name__ == "ovphysx"

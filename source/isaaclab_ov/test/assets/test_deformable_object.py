# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""OVPhysX deformable object tests: host-side asset shells first, then real-backend scenes."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

import ovphysx.types  # noqa: F401
import pytest
import torch
import warp as wp
from flaky import flaky
from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import DeformableObject as OvPhysxDeformableObject  # noqa: E402
from isaaclab_ov.assets.deformable_object.deformable_object_data import DeformableObjectData  # noqa: E402
from isaaclab_ov.assets.deformable_object.kernels import vec6f  # noqa: E402
from isaaclab_ov.assets.deformable_object.views import OvPhysxDeformableBodyView  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager  # noqa: E402
from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg  # noqa: E402
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg  # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObjectCfg  # noqa: E402
from isaaclab.cloner import path, query
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from ..deformable_utils import (  # noqa: E402
    pre_tetrahedralized_deformable_spawn_cfg,
    pretriangulated_surface_deformable_spawn_cfg,
)

wp.init()


@configclass
class HeterogeneousMixedDeformableRigidSceneCfg(InteractiveSceneCfg):
    """Interactive scene configuration with two rigid variants and a deformable."""

    deformable: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=pre_tetrahedralized_deformable_spawn_cfg(),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    shape: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shape",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
                sim_utils.SphereCfg(radius=0.05),
            ],
            rigid_props=PhysxRigidBodyCfg(disable_gravity=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            random_choice=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 1.0)),
    )


def _ovphysx_sim_context(device: str, *, gravity_enabled: bool = True):
    """Build a kitless OVPhysX simulation context."""
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=0.01, gravity=gravity)
    return build_simulation_context(device=device, sim_cfg=sim_cfg, auto_add_lighting=True)


def _generate_deformable_scene(
    spawn: sim_utils.SpawnerCfg,
    num_objects: int = 2,
    height: float = 1.0,
    initial_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> DeformableObject:
    """Create independently authored deformables beneath matching parent prims."""
    for index in range(num_objects):
        sim_utils.create_prim(f"/World/Table_{index}", "Xform", translation=(index * 1.0, 0.0, height))
    cfg = DeformableObjectCfg(
        prim_path="/World/Table_[^/]*/Object",
        spawn=spawn,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height), rot=initial_rot),
    )
    return DeformableObject(cfg=cfg)


def _assert_finite_deformable_state(deformable: DeformableObject) -> None:
    """Assert finite nodal and derived root state."""
    assert torch.isfinite(deformable.data.nodal_state_w.torch).all()
    assert torch.isfinite(deformable.data.root_pos_w.torch).all()
    assert torch.isfinite(deformable.data.root_vel_w.torch).all()


def _canonical_connectivity(connectivity: torch.Tensor) -> list[set[tuple[int, ...]]]:
    """Return unordered elements with each element's vertex indices sorted."""
    return [{tuple(sorted(element)) for element in body} for body in connectivity.cpu().tolist()]


def _assert_rest_positions_match_authored(
    rest_positions: torch.Tensor, authored_points: torch.Tensor, prim_paths: list[str]
) -> None:
    """Assert each body contains the authored local rest points transformed into world space."""
    assert rest_positions.shape == (rest_positions.shape[0], authored_points.shape[0], 3)
    assert torch.is_floating_point(rest_positions)
    assert torch.isfinite(rest_positions).all()
    assert len(prim_paths) == rest_positions.shape[0]
    stage = sim_utils.get_current_stage()
    xform_cache = UsdGeom.XformCache()
    for body_points, prim_path in zip(rest_positions, prim_paths):
        local_to_world = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path))
        expected = torch.tensor(
            [tuple(local_to_world.Transform(Gf.Vec3d(*point.tolist()))) for point in authored_points.cpu()],
            dtype=rest_positions.dtype,
            device=rest_positions.device,
        )
        distances = torch.cdist(body_points, expected)
        torch.testing.assert_close(
            distances.min(dim=0).values, torch.zeros(expected.shape[0], device=expected.device), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            distances.min(dim=1).values, torch.zeros(expected.shape[0], device=expected.device), atol=1e-6, rtol=0.0
        )


##
# Host-side units on asset shells over a fake body view. They need no simulation and run first.
##


@pytest.fixture
def host_shell(monkeypatch):
    """Run an asset shell on CPU with a scene-data backend, without a simulation."""
    monkeypatch.setattr(OvPhysxManager, "_scene_data_backend", OvPhysxSceneDataBackend())
    with wp.ScopedDevice("cpu"):
        yield


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
        assert indices is None or indices.is_contiguous  # Required by the OVPhysX DLPack binding.
        self.last_indices = indices
        if tensor_type in (TT.DEFORMABLE_SIM_NODAL_POSITION, TT.SURFACE_DEFORMABLE_SIM_POSITION):
            self.position_write_count += 1
        elif tensor_type in (TT.DEFORMABLE_SIM_NODAL_VELOCITY, TT.SURFACE_DEFORMABLE_SIM_VELOCITY):
            self.velocity_write_count += 1
        elif tensor_type != TT.DEFORMABLE_SIM_KINEMATIC_TARGET:
            raise AssertionError(f"Unexpected tensor write: {tensor_type}")


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
) -> OvPhysxDeformableObject:
    asset = object.__new__(OvPhysxDeformableObject)
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
    asset._material_physx_view = None
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


@pytest.mark.usefixtures("host_shell")
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

    # Prime the derived buffers at the current timestamp so the write itself must invalidate them.
    asset.data.nodal_state_w
    asset.data.root_pos_w
    asset.data.root_vel_w

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


@pytest.mark.usefixtures("host_shell")
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


@pytest.mark.usefixtures("host_shell")
@pytest.mark.parametrize(
    ("property_name", "write_method_name", "simulator_attribute", "command_value"),
    [
        ("nodal_pos_w", "write_nodal_pos_to_sim_index", "positions", 100.0),
        ("nodal_vel_w", "write_nodal_velocity_to_sim_index", "velocities", -100.0),
    ],
)
@pytest.mark.parametrize("env_ids", [slice(1, 2), slice(None, None, 2)], ids=["contiguous", "strided"])
def test_indexed_partial_write_preserves_retained_aliased_slice(
    property_name: str, write_method_name: str, simulator_attribute: str, command_value: float, env_ids: slice
) -> None:
    """A retained selected slice survives hydration of stale unselected rows."""
    asset = _make_asset_shell(deformable_type="volume", num_instances=3, num_vertices=2)
    initial = torch.full((3, 2, 3), -1.0, device=asset.device)
    setattr(asset.root_view, simulator_attribute, wp.from_torch(initial.contiguous(), dtype=wp.float32))
    retained = getattr(asset.data, property_name)

    latest = torch.arange(18, dtype=torch.float32, device=asset.device).reshape(3, 2, 3)
    setattr(asset.root_view, simulator_attribute, wp.from_torch(latest.contiguous(), dtype=wp.float32))
    asset.update(0.1)
    selected = retained.torch[env_ids]
    selected.fill_(command_value)

    getattr(asset, write_method_name)(selected, env_ids=env_ids)
    # Only the selected environments are written back to the simulator.
    assert asset.root_view.last_indices.numpy().tolist() == list(range(3)[env_ids])

    expected = latest.clone()
    expected[env_ids].fill_(command_value)
    assert getattr(asset.data, property_name) is retained
    torch.testing.assert_close(retained.torch, expected)


@pytest.mark.usefixtures("host_shell")
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


@pytest.mark.usefixtures("host_shell")
def test_malformed_state_write_fails_before_mutating_or_writing():
    asset = _make_asset_shell(deformable_type="volume", num_instances=2, num_vertices=4)
    original_positions = asset.data.nodal_pos_w.torch.clone()
    original_velocities = asset.data.nodal_vel_w.torch.clone()
    malformed_state = torch.ones((1, 4, 7), device=asset.device)

    with pytest.raises(AssertionError, match="nodal_state.*Shape mismatch"):
        asset.write_nodal_state_to_sim_index(malformed_state, env_ids=[1])

    torch.testing.assert_close(asset.data.nodal_pos_w.torch, original_positions)
    torch.testing.assert_close(asset.data.nodal_vel_w.torch, original_velocities)
    assert asset.root_view.position_write_count == 0
    assert asset.root_view.velocity_write_count == 0


@pytest.mark.usefixtures("host_shell")
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


@pytest.mark.usefixtures("host_shell")
def test_surface_debug_visualization_uses_below_ground_sentinel():
    asset = _make_asset_shell(deformable_type="surface")
    asset.target_visualizer = _FakeVisualizer()

    asset._debug_vis_callback(None)

    torch.testing.assert_close(asset.target_visualizer.positions, torch.tensor([[0.0, 0.0, -10.0]]))


class _MixedTopologyBinding:
    """Padded simulation connectivity with five nodes in one body and four in another."""

    _SHAPES = {
        TT.DEFORMABLE_SIM_NODAL_POSITION: (2, 5, 3),
        TT.DEFORMABLE_SIM_ELEMENT_INDICES: (2, 2, 4),
        TT.DEFORMABLE_COLLISION_ELEMENT_INDICES: (2, 6, 4),
    }

    def __init__(self, tensor_type: int):
        self._tensor_type = tensor_type
        self.shape = self._SHAPES[tensor_type]
        float_dtype = tensor_type == TT.DEFORMABLE_SIM_NODAL_POSITION
        self.dtype = SimpleNamespace(code=2 if float_dtype else 0, bits=32, lanes=1)
        self.count = self.shape[0]
        self.prim_paths = [f"/World/env_{index}/Soft" for index in range(self.count)]

    def read(self, values: wp.array) -> None:
        if self._tensor_type == TT.DEFORMABLE_SIM_ELEMENT_INDICES:
            connectivity = [[[0, 1, 2, 3], [1, 2, 3, 4]], [[0, 1, 2, 3], [0, 0, 0, 0]]]
            wp.copy(values, wp.array(connectivity, dtype=wp.int32, device="cpu"))
        elif self._tensor_type == TT.DEFORMABLE_COLLISION_ELEMENT_INDICES:
            wp.copy(values, wp.full(self.shape, value=4, dtype=wp.int32, device="cpu"))


@pytest.mark.usefixtures("host_shell")
def test_volume_view_rejects_mixed_simulation_node_counts():
    physx = SimpleNamespace(
        create_tensor_binding=lambda *, tensor_type, pattern=None: _MixedTopologyBinding(tensor_type)
    )
    with pytest.raises(ValueError, match=r"uniform simulation-node counts.*\[5, 4\]"):
        OvPhysxDeformableBodyView(
            physx,
            pattern="/World/env_*/Soft",
            device="cpu",
            tensor_types=list(_MixedTopologyBinding._SHAPES),
            eager=True,
            simulation_nodal_position_type=TT.DEFORMABLE_SIM_NODAL_POSITION,
            simulation_element_indices_type=TT.DEFORMABLE_SIM_ELEMENT_INDICES,
            collision_element_indices_type=TT.DEFORMABLE_COLLISION_ELEMENT_INDICES,
        )


##
# Real-backend tests.
##


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
def test_initialization_with_shared_material():
    """Test volume deformable initialization with a shared material, and public buffer shapes.

    The shared absolute material path only textually prefixes a sibling asset path. The no-material case is covered
    by :func:`test_set_nodal_state_with_applied_transform`.
    """
    num_objects, material_path = 2, "/World/Table_0/ObjectSiblingMaterial"
    with _ovphysx_sim_context(device="cuda:0") as sim:
        deformable = _generate_deformable_scene(
            pre_tetrahedralized_deformable_spawn_cfg(material_path=material_path), num_objects=num_objects
        )
        # A same-named material under the other table must not be matched by prefix expansion.
        distractor_cfg = PhysxDeformableBodyMaterialCfg()
        distractor_cfg.func("/World/Table_1/ObjectSiblingMaterial", distractor_cfg)

        assert sys.getrefcount(deformable) < 10
        sim.reset()

        assert deformable.is_initialized
        assert deformable.num_instances == num_objects
        assert deformable.num_bodies == 1
        assert deformable.root_view.count == num_objects
        assert deformable.material_physx_view is not None
        assert deformable.material_physx_view.count == 1
        assert deformable.data.nodal_state_w.torch.shape == (
            num_objects,
            deformable.max_sim_vertices_per_body,
            6,
        )
        assert deformable.data.nodal_kinematic_target is not None
        assert deformable.data.nodal_kinematic_target.torch.shape == (
            num_objects,
            deformable.max_sim_vertices_per_body,
            4,
        )
        assert deformable.data.root_pos_w.torch.shape == (num_objects, 3)
        assert deformable.data.root_vel_w.torch.shape == (num_objects, 3)

        deformable._invalidate_initialize_callback(None)
        assert deformable._root_physx_view is None
        assert deformable._material_physx_view is None


@pytest.mark.parametrize("deformable_type", ["volume", "surface"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
def test_mesh_spawner_deformable_fragment_slots(deformable_type: str):
    """The mesh spawner's deformable fragment slots author a deformable that OVPhysX simulates."""
    from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

    from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg

    body_cfg = OmniPhysicsDeformableBodyCfg(mass=0.5)
    if deformable_type == "volume":
        pytest.importorskip("pytetwild", reason="volume deformables are tetrahedralized with pytetwild")
        spawn = sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2), volume_deformable_props=body_cfg, physics_material=PhysxDeformableBodyMaterialCfg()
        )
    else:
        spawn = sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2), surface_deformable_props=body_cfg, physics_material=PhysxSurfaceDeformableBodyMaterialCfg()
        )
    with _ovphysx_sim_context(device="cuda:0") as sim:
        deformable = _generate_deformable_scene(spawn)

        sim.reset()

        assert deformable.is_initialized
        assert deformable._deformable_type == deformable_type
        assert deformable.num_instances == 2
        stage = sim_utils.get_current_stage()
        for index in range(2):
            body = stage.GetPrimAtPath(f"/World/Table_{index}/Object")
            assert "OmniPhysicsDeformableBodyAPI" in body.GetPrimTypeInfo().GetAppliedAPISchemas()
            assert body.GetAttribute("omniphysics:mass").Get() == pytest.approx(0.5)
        for _ in range(5):
            sim.step()
            deformable.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)


def test_initialization_on_device_cpu():
    """Test that OVPhysX deformable initialization rejects a CPU simulation.

    CPU and CUDA OVPhysX simulations can follow each other in one process, so this runs beside the CUDA tests.
    """
    message = "OVPhysX deformable tensors require a CUDA simulation device; received 'cpu'."
    with _ovphysx_sim_context(device="cpu") as sim:
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg(), num_objects=5)
        assert sys.getrefcount(deformable) < 10
        with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
            sim.reset()
        assert deformable.is_initialized is False


@flaky(max_runs=3, min_passes=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
def test_set_nodal_state_with_applied_transform():
    """Test combined nodal state writes after applying rigid transforms, on deformables without a material."""
    with _ovphysx_sim_context(device="cuda:0", gravity_enabled=False) as sim:
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg(material_path=None))
        sim.reset()
        assert deformable.material_physx_view is None

        for _ in range(2):
            nodal_state = deformable.data.default_nodal_state_w.torch.clone()
            mean_nodal_pos_default = nodal_state[..., :3].mean(dim=1)

            pos_w = 0.5 * torch.rand(deformable.num_instances, 3, device=sim.device)
            pos_w[:, 2] += 0.5
            quat_w = math_utils.random_orientation(deformable.num_instances, device=sim.device)

            nodal_state[..., :3] = deformable.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
            mean_nodal_pos_init = nodal_state[..., :3].mean(dim=1)
            torch.testing.assert_close(mean_nodal_pos_init, mean_nodal_pos_default + pos_w, rtol=1e-5, atol=1e-5)

            deformable.write_nodal_state_to_sim_index(nodal_state)
            deformable.reset()

            for _ in range(10):
                sim.step()
                deformable.update(sim.cfg.dt)

            torch.testing.assert_close(deformable.data.root_pos_w.torch, mean_nodal_pos_init, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
def test_set_kinematic_targets():
    """Test pinning one volume deformable while another falls under gravity."""
    with _ovphysx_sim_context(device="cuda:0", gravity_enabled=True) as sim:
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg(), num_objects=2, height=1.0)
        sim.reset()

        nodal_kinematic_targets = wp.to_torch(
            deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_KINEMATIC_TARGET)
        ).clone()

        for _ in range(2):
            deformable.write_nodal_state_to_sim_index(deformable.data.default_nodal_state_w.torch)
            default_root_pos = deformable.data.default_nodal_state_w.torch[..., :3].mean(dim=1)
            deformable.reset()

            nodal_kinematic_targets[1:, :, 3] = 1.0
            nodal_kinematic_targets[0, :, 3] = 0.0
            nodal_kinematic_targets[0, :, :3] = deformable.data.default_nodal_state_w.torch[0, :, :3]
            deformable.write_nodal_kinematic_target_to_sim_index(
                nodal_kinematic_targets[0:1], env_ids=torch.tensor([0], device=sim.device)
            )

            for _ in range(10):
                sim.step()
                deformable.update(sim.cfg.dt)

                torch.testing.assert_close(
                    deformable.data.nodal_pos_w.torch[0],
                    nodal_kinematic_targets[0, :, :3],
                    rtol=1e-5,
                    atol=1e-5,
                )
                assert torch.all(deformable.data.root_pos_w.torch[1:, 2] < default_root_pos[1:, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
def test_volume_deformable_reads_writes_targets_materials_and_steps():
    """Exercise authored volume state, topology, targets, materials, and stepping."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        deformable = _generate_deformable_scene(pre_tetrahedralized_deformable_spawn_cfg())

        sim.reset()

        assert deformable.is_initialized
        assert deformable.num_instances == 2
        assert deformable.num_bodies == 1
        assert deformable.root_view.count == 2
        assert deformable.max_sim_vertices_per_body == 5
        assert deformable.max_sim_elements_per_body == 2
        assert deformable.max_collision_elements_per_body == 2
        assert deformable.max_collision_vertices_per_body == 5

        nodal_state = deformable.data.nodal_state_w.torch
        nodal_pos = deformable.data.nodal_pos_w.torch
        nodal_vel = deformable.data.nodal_vel_w.torch
        assert nodal_state.shape == (2, 5, 6)
        assert deformable.data.default_nodal_state_w.torch.shape == (2, 5, 6)
        assert deformable.data.root_pos_w.torch.shape == (2, 3)
        assert deformable.data.root_vel_w.torch.shape == (2, 3)
        rest_positions = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_REST_NODAL_POSITION))
        _assert_rest_positions_match_authored(
            rest_positions,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.2],
                    [0.2, 0.2, 0.2],
                ],
                device=rest_positions.device,
            ),
            deformable.root_view.prim_paths,
        )
        torch.testing.assert_close(deformable.data.root_pos_w.torch, nodal_pos.mean(dim=1))
        torch.testing.assert_close(deformable.data.root_vel_w.torch, nodal_vel.mean(dim=1))

        element_indices = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_ELEMENT_INDICES))
        collision_indices = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_COLLISION_ELEMENT_INDICES))
        assert element_indices.shape == (2, 2, 4)
        assert collision_indices.shape == (2, 2, 4)
        assert element_indices.dtype == torch.int32
        assert collision_indices.dtype == torch.int32
        assert torch.all((element_indices >= 0) & (element_indices < 5))
        assert torch.all((collision_indices >= 0) & (collision_indices < 5))
        expected_tetrahedra = {(0, 1, 2, 3), (1, 2, 3, 4)}
        assert all(elements == expected_tetrahedra for elements in _canonical_connectivity(element_indices))
        assert all(elements == expected_tetrahedra for elements in _canonical_connectivity(collision_indices))

        updated_pos = nodal_pos[1:2].clone()
        updated_pos[..., 0] += 0.025
        deformable.write_nodal_pos_to_sim_index(updated_pos, env_ids=torch.tensor([1], device=sim.device))
        readback_pos = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_NODAL_POSITION))
        torch.testing.assert_close(readback_pos[0], nodal_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[0], rtol=1e-5, atol=1e-5)

        updated_vel = nodal_vel[0:1].clone()
        updated_vel[..., 1] = 0.1
        deformable.write_nodal_velocity_to_sim_index(updated_vel, env_ids=torch.tensor([0]))
        readback_vel = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_NODAL_VELOCITY))
        torch.testing.assert_close(readback_vel[0], updated_vel[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_vel[1], nodal_vel[1], rtol=1e-5, atol=1e-5)

        targets = deformable.data.nodal_kinematic_target
        assert targets is not None
        assert targets.torch.shape == (2, 5, 4)
        torch.testing.assert_close(targets.torch[..., 3], torch.ones_like(targets.torch[..., 3]))
        updated_targets = targets.torch[1:2].clone()
        updated_targets[..., :3] = readback_pos[1:2] + torch.tensor([0.0, 0.0, 0.03], device=sim.device)
        updated_targets[..., 3] = 0.0
        deformable.write_nodal_kinematic_target_to_sim_index(
            updated_targets, env_ids=torch.tensor([1], device=sim.device)
        )
        readback_targets = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_KINEMATIC_TARGET))
        torch.testing.assert_close(readback_targets[0, :, 3], torch.ones_like(readback_targets[0, :, 3]))
        torch.testing.assert_close(readback_targets[1], updated_targets[0], rtol=1e-5, atol=1e-5)

        material_view = deformable.material_physx_view
        assert material_view is not None
        assert material_view.count == 2
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION)), torch.full((2,), 0.5)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS)), torch.full((2,), 1000.0)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_POISSONS_RATIO)), torch.full((2,), 0.3)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_ELASTICITY_DAMPING)), torch.full((2,), 0.005)
        )

        updated_youngs = torch.tensor([1000.0, 1500.0])
        material_view.set_attribute(
            TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS,
            wp.from_torch(updated_youngs),
            # OvPhysX CPU-native material bindings require host-resident indices.
            indices=wp.array([1], dtype=wp.int32, device="cpu"),
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS)), updated_youngs.cpu()
        )

        for _ in range(5):
            sim.step()
            deformable.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)

        # A forced re-warm replaces the attached stage, so the deformable bindings are rebuilt.
        original_view = deformable.root_view
        original_binding = original_view.binding_for(TT.DEFORMABLE_SIM_NODAL_POSITION)
        OvPhysxManager._warmup_done = False
        sim.reset()

        assert deformable.is_initialized
        assert deformable.root_view is not original_view
        assert deformable.root_view.binding_for(TT.DEFORMABLE_SIM_NODAL_POSITION) is not original_binding
        _assert_finite_deformable_state(deformable)
        sim.step()
        deformable.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
def test_surface_deformable_reads_writes_materials_and_steps():
    """Exercise authored surface state, topology, materials, and stepping."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        deformable = _generate_deformable_scene(pretriangulated_surface_deformable_spawn_cfg())

        sim.reset()

        assert deformable.is_initialized
        assert deformable.num_instances == 2
        assert deformable.root_view.count == 2
        assert deformable.max_sim_vertices_per_body == 4
        assert deformable.max_sim_elements_per_body == 2
        assert deformable.max_collision_elements_per_body == 0
        assert deformable.max_collision_vertices_per_body == 0
        assert deformable.data.nodal_state_w.torch.shape == (2, 4, 6)
        assert deformable.data.root_pos_w.torch.shape == (2, 3)
        assert deformable.data.root_vel_w.torch.shape == (2, 3)
        rest_positions = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_REST_POSITION))
        _assert_rest_positions_match_authored(
            rest_positions,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.2, 0.2, 0.0],
                    [0.0, 0.2, 0.0],
                ],
                device=rest_positions.device,
            ),
            deformable.root_view.prim_paths,
        )

        element_indices = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES))
        assert element_indices.shape == (2, 2, 3)
        assert element_indices.dtype == torch.int32
        assert torch.all((element_indices >= 0) & (element_indices < 4))
        expected_triangles = {(0, 1, 2), (0, 2, 3)}
        assert all(elements == expected_triangles for elements in _canonical_connectivity(element_indices))

        assert deformable.data.nodal_kinematic_target is None
        dummy_targets = torch.zeros((2, 4, 4), device=sim.device)
        with pytest.raises(ValueError, match="Kinematic targets can only be set for volume deformable bodies"):
            deformable.write_nodal_kinematic_target_to_sim_index(dummy_targets)

        nodal_pos = deformable.data.nodal_pos_w.torch
        updated_pos = nodal_pos[1:2].clone()
        updated_pos[..., 0] += 0.025
        deformable.write_nodal_pos_to_sim_index(updated_pos, env_ids=torch.tensor([1], device=sim.device))
        readback_pos = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_SIM_POSITION))
        torch.testing.assert_close(readback_pos[0], nodal_pos[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_pos[1], updated_pos[0], rtol=1e-5, atol=1e-5)

        nodal_vel = deformable.data.nodal_vel_w.torch
        updated_vel = nodal_vel[0:1].clone()
        updated_vel[..., 1] = 0.1
        deformable.write_nodal_velocity_to_sim_index(updated_vel, env_ids=torch.tensor([0]))
        readback_vel = wp.to_torch(deformable.root_view.get_attribute(TT.SURFACE_DEFORMABLE_SIM_VELOCITY))
        torch.testing.assert_close(readback_vel[0], updated_vel[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(readback_vel[1], nodal_vel[1], rtol=1e-5, atol=1e-5)

        material_view = deformable.material_physx_view
        assert material_view is not None
        assert material_view.count == 2
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_DYNAMIC_FRICTION)), torch.full((2,), 0.4)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_YOUNGS_MODULUS)), torch.full((2,), 2000.0)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_POISSONS_RATIO)), torch.full((2,), 0.25)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_ELASTICITY_DAMPING)), torch.full((2,), 0.03)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_BENDING_STIFFNESS)), torch.full((2,), 0.6)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_THICKNESS)), torch.full((2,), 0.02)
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_BENDING_DAMPING)), torch.full((2,), 0.04)
        )

        updated_bending_damping = torch.tensor([0.08, 0.04])
        material_view.set_attribute(
            TT.DEFORMABLE_MATERIAL_BENDING_DAMPING,
            wp.from_torch(updated_bending_damping),
            # OvPhysX CPU-native material bindings require host-resident indices.
            indices=wp.array([0], dtype=wp.int32, device="cpu"),
        )
        torch.testing.assert_close(
            wp.to_torch(material_view.get_attribute(TT.DEFORMABLE_MATERIAL_BENDING_DAMPING)),
            updated_bending_damping.cpu(),
        )

        for _ in range(5):
            sim.step()
            deformable.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
def test_heterogeneous_mixed_deformable_rigid_scene_materializes_missing_targets():
    """Materialize missing rigid targets beside full-stage deformable clones without duplicates."""
    with _ovphysx_sim_context(device="cuda:0") as sim:
        num_envs = 4
        scene = InteractiveScene(
            HeterogeneousMixedDeformableRigidSceneCfg(num_envs=num_envs, env_spacing=1.0, lazy_sensor_update=False)
        )
        plan = sim.get_clone_plan()
        source_paths = path.get_asset_prototype_paths(plan)
        shape_ids = path.get_asset_prototypes(plan, scene.cfg.shape.prim_path)
        worlds, starts = query.get_asset_prototype_unique_world_index(plan.topology, shape_ids)
        assert (starts[:, -1] - starts[:, 0]).tolist() == [2, 2]
        assert sorted(worlds) == list(range(num_envs))

        expected_paths = {f"/World/envs/env_{index}/Shape" for index in range(num_envs)}
        shape_paths = {source_paths[index] for index in shape_ids}
        assert shape_paths == {"/World/envs/env_0/Shape", "/World/envs/env_2/Shape"}
        stage = sim_utils.get_current_stage()
        ancestor_path = "/World/envs/env_1/Shape"
        camera_path = f"{ancestor_path}/Camera"
        UsdGeom.Xform.Define(stage, camera_path)
        authored_paths = {path for path in expected_paths if stage.GetPrimAtPath(path).IsValid()}
        assert authored_paths == shape_paths | {ancestor_path}
        authored_deformable_paths = {f"/World/envs/env_{index}/Object/simulation" for index in range(num_envs)}
        deformable_source_paths = {
            f"{source_paths[index]}/simulation"
            for index in path.get_asset_prototypes(plan, scene.cfg.deformable.prim_path)
        }
        assert {
            path for path in authored_deformable_paths if stage.GetPrimAtPath(path).IsValid()
        } == deformable_source_paths

        sim.reset()

        deformable = scene["deformable"]
        shape = scene["shape"]
        assert deformable.num_instances == num_envs
        assert deformable.root_view.count == num_envs, deformable.root_view.prim_paths
        assert deformable.material_physx_view is not None
        assert deformable.material_physx_view.count == num_envs
        assert shape.num_instances == num_envs
        assert shape.root_view.count == num_envs, shape.root_view.prim_paths
        runtime_paths = shape.root_view.prim_paths
        assert set(runtime_paths) == expected_paths
        assert len(runtime_paths) == len(set(runtime_paths)) == num_envs
        assert OvPhysxManager._stage_usda is not None
        layer = Sdf.Layer.CreateAnonymous("materialized.usda")
        assert layer.ImportFromString(OvPhysxManager._stage_usda)
        materialized_stage = Usd.Stage.Open(layer)
        assert materialized_stage.GetPrimAtPath(camera_path).IsValid()
        assert all(materialized_stage.GetPrimAtPath(path).IsValid() for path in authored_deformable_paths)

        sim.step()
        scene.update(sim.cfg.dt)
        _assert_finite_deformable_state(deformable)
        assert torch.isfinite(shape.data.root_pos_w.torch).all()

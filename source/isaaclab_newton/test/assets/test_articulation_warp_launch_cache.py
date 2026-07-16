# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp launch handling at the Newton Isaac Lab actuator boundary."""

from types import SimpleNamespace

import isaaclab_newton.assets.articulation.articulation as articulation_module
import isaaclab_newton.assets.rigid_object.rigid_object as rigid_object_module
import pytest
import torch
import warp as wp
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject
from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationContext
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import WarpLaunchCache


class _AllocatingActuator:
    """Minimal explicit actuator that returns new Torch allocations per call."""

    def __init__(self, num_envs: int, num_joints: int, device: str):
        self.is_implicit_model = False
        self.joint_indices = slice(None)
        self.gear_ratio = None
        self.velocity_limit = torch.full((num_envs, num_joints), 10.0, device=device)
        self.computed_effort = torch.zeros((num_envs, num_joints), device=device)
        self.applied_effort = torch.zeros_like(self.computed_effort)
        self.output_pointers: list[tuple[int, int, int, int]] = []

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        del joint_pos, joint_vel
        self.computed_effort = control_action.joint_efforts + 1.0
        self.applied_effort = self.computed_effort * 2.0
        if self.output_pointers:
            self.gear_ratio = self.computed_effort + 10.0
            control_action.joint_positions = self.applied_effort + 100.0
        else:
            self.gear_ratio = None
            control_action.joint_positions = None
        self.velocity_limit = self.computed_effort + 20.0
        self.output_pointers.append(
            (
                self.computed_effort.data_ptr(),
                self.applied_effort.data_ptr(),
                0 if self.gear_ratio is None else self.gear_ratio.data_ptr(),
                self.velocity_limit.data_ptr(),
            )
        )
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action


class _AllocatingImplicitActuator:
    """Minimal implicit actuator with persistent targets and allocating telemetry."""

    is_implicit_model = True

    def __init__(self, num_envs: int, num_joints: int, device: str):
        self.joint_indices = slice(None)
        self.gear_ratio = None
        self.velocity_limit = torch.full((num_envs, num_joints), 10.0, device=device)
        self.computed_effort = torch.zeros((num_envs, num_joints), device=device)
        self.applied_effort = torch.zeros_like(self.computed_effort)
        self.output_pointers: list[tuple[int, int]] = []

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        del joint_pos, joint_vel
        self.computed_effort = control_action.joint_efforts + 1.0
        self.applied_effort = self.computed_effort * 2.0
        self.output_pointers.append((self.computed_effort.data_ptr(), self.applied_effort.data_ptr()))
        return control_action


def _make_articulation_stub(
    actuator: _AllocatingActuator | _AllocatingImplicitActuator,
    num_envs: int,
    num_joints: int,
    device: str,
) -> tuple[SimpleNamespace, torch.Tensor]:
    """Create the minimum articulation state needed by the actuator publication path."""
    shape = (num_envs, num_joints)
    articulation = SimpleNamespace(device=device, num_instances=num_envs)
    articulation._ALL_JOINT_INDICES = wp.array(range(num_joints), dtype=wp.int32, device=device)
    articulation._joint_pos_target_sim = wp.zeros(shape, dtype=wp.float32, device=device)
    articulation._joint_vel_target_sim = wp.zeros(shape, dtype=wp.float32, device=device)
    articulation._joint_effort_target_sim = wp.zeros(shape, dtype=wp.float32, device=device)
    articulation._warp_launch = WarpLaunchCache(mode="replay", debug=True, device=device)
    articulation.actuators = {"test": actuator}

    joint_effort_target = torch.ones(shape, device=device)
    articulation._data = SimpleNamespace(
        joint_pos_target=SimpleNamespace(torch=torch.zeros(shape, device=device)),
        joint_vel_target=SimpleNamespace(torch=torch.zeros(shape, device=device)),
        joint_effort_target=SimpleNamespace(torch=joint_effort_target),
        joint_pos=SimpleNamespace(torch=torch.zeros(shape, device=device)),
        joint_vel=SimpleNamespace(torch=torch.zeros(shape, device=device)),
        computed_torque=wp.zeros(shape, dtype=wp.float32, device=device),
        applied_torque=wp.zeros(shape, dtype=wp.float32, device=device),
        gear_ratio=wp.zeros(shape, dtype=wp.float32, device=device),
        soft_joint_vel_limits=wp.zeros(shape, dtype=wp.float32, device=device),
    )
    return articulation, joint_effort_target


@pytest.fixture(scope="module")
def cuda_device() -> str:
    """Return a CUDA device or skip the module when CUDA is unavailable."""
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required for Warp launch replay tests.")
    return "cuda:0"


def test_newton_launch_cache_is_disabled_by_default():
    """Newton should preserve eager actuator-boundary launches unless opted in."""
    assert NewtonCfg().use_warp_launch_cache is False


@pytest.mark.parametrize(
    ("asset_cls", "asset_module"),
    [(Articulation, articulation_module), (RigidObject, rigid_object_module)],
)
def test_asset_launch_cache_uses_sim_device_before_asset_initialization(
    asset_cls: type,
    asset_module,
    monkeypatch: pytest.MonkeyPatch,
):
    """Asset construction should not read ``asset.device`` before the ready callback sets it."""
    sim_context = SimpleNamespace(
        cfg=SimpleNamespace(physics=SimpleNamespace(use_warp_launch_cache=True)),
        device="cpu",
    )
    monkeypatch.setattr(SimulationContext, "instance", classmethod(lambda cls: sim_context))
    monkeypatch.setattr(asset_cls.__mro__[1], "__init__", lambda self, cfg: setattr(self, "cfg", cfg))
    monkeypatch.setattr(asset_cls, "_clear_callbacks", lambda self: None)
    monkeypatch.setattr(asset_module, "has_kit", lambda: False)
    monkeypatch.setattr(asset_module, "queue_newton_physics_replication", lambda cfg: None)

    asset = object.__new__(asset_cls)
    asset_cls.__init__(asset, SimpleNamespace())

    assert not hasattr(asset, "_device")
    assert asset._warp_launch._device == wp.get_device("cpu")
    assert asset._warp_launch._mode == "replay"
    del asset


def test_actuator_publication_accepts_reallocated_torch_outputs(cuda_device: str):
    """Eager publication should use current actuator tensors after their pointers change."""
    num_envs = 8
    num_joints = 3
    actuator = _AllocatingActuator(num_envs, num_joints, cuda_device)
    articulation, joint_effort_target = _make_articulation_stub(actuator, num_envs, num_joints, cuda_device)
    shape = (num_envs, num_joints)

    Articulation._apply_actuator_model(articulation)
    joint_effort_target.fill_(3.0)
    Articulation._apply_actuator_model(articulation)
    wp.synchronize_device(cuda_device)

    assert actuator.output_pointers[0] != actuator.output_pointers[1]
    assert len(articulation._warp_launch._entries) == 0
    torch.testing.assert_close(
        wp.to_torch(articulation._joint_pos_target_sim), torch.full(shape, 108.0, device=cuda_device)
    )
    torch.testing.assert_close(
        wp.to_torch(articulation._joint_effort_target_sim), torch.full(shape, 8.0, device=cuda_device)
    )
    torch.testing.assert_close(
        wp.to_torch(articulation._data.computed_torque), torch.full(shape, 4.0, device=cuda_device)
    )
    torch.testing.assert_close(
        wp.to_torch(articulation._data.applied_torque), torch.full(shape, 8.0, device=cuda_device)
    )
    torch.testing.assert_close(wp.to_torch(articulation._data.gear_ratio), torch.full(shape, 14.0, device=cuda_device))
    torch.testing.assert_close(
        wp.to_torch(articulation._data.soft_joint_vel_limits), torch.full(shape, 24.0, device=cuda_device)
    )


def test_full_slice_implicit_targets_use_recorded_launch(cuda_device: str):
    """Persistent full-group implicit targets should replay while allocating telemetry stays eager."""
    num_envs = 8
    num_joints = 3
    shape = (num_envs, num_joints)
    actuator = _AllocatingImplicitActuator(num_envs, num_joints, cuda_device)
    articulation, joint_effort_target = _make_articulation_stub(actuator, num_envs, num_joints, cuda_device)

    Articulation._apply_actuator_model(articulation)
    joint_effort_target.fill_(3.0)
    Articulation._apply_actuator_model(articulation)
    wp.synchronize_device(cuda_device)

    assert actuator.output_pointers[0] != actuator.output_pointers[1]
    assert len(articulation._warp_launch._entries) == 1
    torch.testing.assert_close(
        wp.to_torch(articulation._joint_effort_target_sim), torch.full(shape, 3.0, device=cuda_device)
    )
    torch.testing.assert_close(
        wp.to_torch(articulation._data.computed_torque), torch.full(shape, 4.0, device=cuda_device)
    )
    torch.testing.assert_close(
        wp.to_torch(articulation._data.applied_torque), torch.full(shape, 8.0, device=cuda_device)
    )


@pytest.mark.parametrize("newton_actuator_path", [False, True])
def test_joint_target_reorder_uses_recorded_launch(cuda_device: str, newton_actuator_path: bool):
    """Persistent Lab and Newton joint-target reorders should replay current in-place values."""
    num_envs = 4
    num_joints = 3
    shape = (num_envs, num_joints)
    source_effort = wp.ones(shape, dtype=wp.float32, device=cuda_device)
    source_pos = wp.full(shape, 2.0, dtype=wp.float32, device=cuda_device)
    source_vel = wp.full(shape, 3.0, dtype=wp.float32, device=cuda_device)
    backend_effort = wp.zeros(shape, dtype=wp.float32, device=cuda_device)
    backend_pos = wp.zeros(shape, dtype=wp.float32, device=cuda_device)
    backend_vel = wp.zeros(shape, dtype=wp.float32, device=cuda_device)
    backend_act = wp.zeros(shape, dtype=wp.float32, device=cuda_device)
    backend_to_user = wp.array([2, 0, 1], dtype=wp.int32, device=cuda_device)

    articulation = SimpleNamespace(
        _has_newton_actuators=newton_actuator_path,
        _has_implicit_actuators=True,
        _instantaneous_wrench_composer=SimpleNamespace(active=False),
        _permanent_wrench_composer=SimpleNamespace(active=False),
        _joint_effort_target_sim=source_effort,
        _joint_pos_target_sim=source_pos,
        _joint_vel_target_sim=source_vel,
        _joint_backend_to_user_map=lambda: backend_to_user,
        _apply_actuator_model=lambda: None,
        _warp_launch=WarpLaunchCache(mode="replay", debug=True, device=cuda_device),
        _data=SimpleNamespace(
            _joint_effort_target=source_effort,
            _joint_pos_target=source_pos,
            _joint_vel_target=source_vel,
        ),
        data=SimpleNamespace(
            _sim_bind_joint_effort=backend_effort,
            _sim_bind_joint_position_target=backend_pos,
            _sim_bind_joint_velocity_target=backend_vel,
            _sim_bind_joint_act=backend_act,
        ),
        device=cuda_device,
        num_instances=num_envs,
        num_joints=num_joints,
    )

    Articulation.write_data_to_sim(articulation)
    source_effort.fill_(4.0)
    source_pos.fill_(5.0)
    source_vel.fill_(6.0)
    Articulation.write_data_to_sim(articulation)
    wp.synchronize_device(cuda_device)

    assert len(articulation._warp_launch._entries) == 1
    entry = next(iter(articulation._warp_launch._entries.values()))
    expected_site = "newton_joint_targets" if newton_actuator_path else "lab_joint_targets"
    assert entry.site == expected_site
    torch.testing.assert_close(wp.to_torch(backend_effort), torch.full(shape, 4.0, device=cuda_device))
    torch.testing.assert_close(wp.to_torch(backend_pos), torch.full(shape, 5.0, device=cuda_device))
    torch.testing.assert_close(wp.to_torch(backend_vel), torch.full(shape, 6.0, device=cuda_device))
    if newton_actuator_path:
        torch.testing.assert_close(wp.to_torch(backend_act), torch.full(shape, 4.0, device=cuda_device))

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import types

import pytest
import torch
from isaaclab_contrib.controllers import (
    lee_acceleration_control as acc_mod,
)
from isaaclab_contrib.controllers import (
    lee_position_control as pos_mod,
)
from isaaclab_contrib.controllers import (
    lee_velocity_control as vel_mod,
)
from isaaclab_contrib.controllers.lee_acceleration_control_cfg import LeeAccControllerCfg
from isaaclab_contrib.controllers.lee_position_control_cfg import LeePosControllerCfg
from isaaclab_contrib.controllers.lee_velocity_control_cfg import LeeVelControllerCfg


class _DummyPhysxView:
    """Minimal physx view providing inertia and inv-mass tensors."""

    def __init__(self, num_envs: int, num_bodies: int, device: torch.device):
        self._inertias = torch.eye(3, device=device).repeat(num_envs, num_bodies, 1, 1)
        self._inv_masses = torch.ones((num_envs, num_bodies), device=device)

    def get_inertias(self) -> torch.Tensor:
        return self._inertias

    def get_inv_masses(self) -> torch.Tensor:
        return self._inv_masses


class _DummyRobot:
    """Minimal multirotor stub exposing the attributes used by the controllers."""

    def __init__(self, num_envs: int, num_bodies: int, device: torch.device):
        self.num_bodies = num_bodies
        quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        self.data = types.SimpleNamespace(
            root_link_quat_w=quat_id.repeat(num_envs, 1),
            root_quat_w=quat_id.repeat(num_envs, 1),
            root_pos_w=torch.zeros((num_envs, 3), device=device),
            root_lin_vel_w=torch.zeros((num_envs, 3), device=device),
            root_ang_vel_b=torch.zeros((num_envs, 3), device=device),
            body_link_pos_w=torch.zeros((num_envs, num_bodies, 3), device=device),
            body_link_quat_w=quat_id.repeat(num_envs, num_bodies, 1),
            body_com_pos_b=torch.zeros((num_envs, num_bodies, 3), device=device),
            body_com_quat_b=quat_id.repeat(num_envs, num_bodies, 1),
        )
        self.root_physx_view = _DummyPhysxView(num_envs, num_bodies, device)


class _DummySimCfg:
    """Mock simulation config."""

    def __init__(self):
        self.gravity = (0.0, 0.0, -9.81)


class _DummySimContext:
    """Mock simulation context."""

    def __init__(self):
        self.cfg = _DummySimCfg()


def _patch_aggregate(monkeypatch: pytest.MonkeyPatch, module, num_envs: int, device: torch.device) -> None:
    """Monkeypatch aggregate_inertia_about_robot_com to a deterministic CPU-friendly stub."""

    def _agg(*_args, **_kwargs):
        return (
            torch.ones(num_envs, device=device),  # mass
            torch.eye(3, device=device).repeat(num_envs, 1, 1),  # inertia
            torch.zeros((num_envs, 3, 3), device=device),
        )

    monkeypatch.setattr(module.math_utils, "aggregate_inertia_about_robot_com", _agg)


def _patch_sim_context(monkeypatch: pytest.MonkeyPatch, module) -> None:
    """Monkeypatch SimulationContext.instance() to return a mock."""
    import isaaclab.sim as sim_utils

    def _mock_instance():
        return _DummySimContext()

    monkeypatch.setattr(sim_utils.SimulationContext, "instance", _mock_instance)


def _device_param(device_str: str) -> torch.device:
    """Return the torch.device or skip when CUDA is unavailable."""
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this system")
    return torch.device(device_str)


def _create_vel_cfg() -> LeeVelControllerCfg:
    """Create velocity controller config with required parameters."""
    cfg = LeeVelControllerCfg()
    cfg.K_vel_range = ((2.7, 2.7, 1.3), (3.3, 3.3, 1.7))
    cfg.K_rot_range = ((1.6, 1.6, 0.25), (1.85, 1.85, 0.4))
    cfg.K_angvel_range = ((0.4, 0.4, 0.075), (0.5, 0.5, 0.09))
    cfg.max_inclination_angle_rad = 1.0471975511965976
    cfg.max_yaw_rate = 1.0471975511965976
    return cfg


def _create_pos_cfg() -> LeePosControllerCfg:
    """Create position controller config with required parameters."""
    cfg = LeePosControllerCfg()
    cfg.K_pos_range = ((3.0, 3.0, 2.0), (4.0, 4.0, 2.5))
    cfg.K_rot_range = ((1.6, 1.6, 0.25), (1.85, 1.85, 0.4))
    cfg.K_angvel_range = ((0.4, 0.4, 0.075), (0.5, 0.5, 0.09))
    cfg.max_inclination_angle_rad = 1.0471975511965976
    cfg.max_yaw_rate = 1.0471975511965976
    return cfg


def _create_acc_cfg() -> LeeAccControllerCfg:
    """Create acceleration controller config with required parameters."""
    cfg = LeeAccControllerCfg()
    cfg.K_rot_range = ((1.6, 1.6, 0.25), (1.85, 1.85, 0.4))
    cfg.K_angvel_range = ((0.4, 0.4, 0.075), (0.5, 0.5, 0.09))
    cfg.max_inclination_angle_rad = 1.0471975511965976
    cfg.max_yaw_rate = 1.0471975511965976
    return cfg


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
@pytest.mark.parametrize("num_envs", [1, 2, 8])
@pytest.mark.parametrize("num_bodies", [1, 4])
@pytest.mark.parametrize(
    "controller_cls,cfg_factory,mod_name",
    [
        ("LeeVelController", _create_vel_cfg, vel_mod),
        ("LeePosController", _create_pos_cfg, pos_mod),
        ("LeeAccController", _create_acc_cfg, acc_mod),
    ],
)
def test_lee_controllers_basic(
    monkeypatch: pytest.MonkeyPatch,
    device_str: str,
    num_envs: int,
    num_bodies: int,
    controller_cls: str,
    cfg_factory,
    mod_name,
):
    """Controllers return finite (N, 6) wrench on zero state and counter gravity on +Z.

    Tests various configurations of number of environments and bodies to catch edge cases.
    """
    device = _device_param(device_str)
    _patch_aggregate(monkeypatch, mod_name, num_envs, device)
    _patch_sim_context(monkeypatch, mod_name)
    robot = _DummyRobot(num_envs, num_bodies, device)

    cfg = cfg_factory()
    cfg.randomize_params = False
    controller = getattr(mod_name, controller_cls)(cfg, robot, num_envs=num_envs, device=str(device))

    if controller_cls == "LeeVelController":
        command = torch.zeros((num_envs, 4), device=device)  # vx, vy, vz, yaw_rate
    else:
        command = torch.zeros((num_envs, 3), device=device)  # position or acceleration setpoint

    wrench = controller.compute(command)

    assert wrench.shape == (num_envs, 6), f"Expected shape ({num_envs}, 6), got {wrench.shape}"
    assert torch.isfinite(wrench).all(), "Wrench contains non-finite values"
    assert torch.all(wrench[:, 2] > 0.0), "Body-z force should oppose gravity"


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
@pytest.mark.parametrize("num_envs", [1, 2, 8])
@pytest.mark.parametrize("num_bodies", [1, 4])
def test_lee_vel_randomize_params_within_bounds(
    monkeypatch: pytest.MonkeyPatch, device_str: str, num_envs: int, num_bodies: int
):
    """Randomized gains stay within configured ranges for velocity controller.

    Tests edge cases with single and multiple environments and bodies.
    """
    device = _device_param(device_str)
    _patch_aggregate(monkeypatch, vel_mod, num_envs, device)
    _patch_sim_context(monkeypatch, vel_mod)
    robot = _DummyRobot(num_envs, num_bodies, device)

    cfg = _create_vel_cfg()
    cfg.randomize_params = True
    controller = vel_mod.LeeVelController(cfg, robot, num_envs=num_envs, device=str(device))

    controller.reset_idx(env_ids=None)

    # Ensure tensors are on the correct device
    K_vel_min = torch.tensor(cfg.K_vel_range[0], device=device, dtype=torch.float32)
    K_vel_max = torch.tensor(cfg.K_vel_range[1], device=device, dtype=torch.float32)

    # Move controller gains to same device if needed
    K_vel_current = controller.K_vel_current.to(device)

    assert K_vel_current.shape == (num_envs, 3), f"Expected shape ({num_envs}, 3), got {K_vel_current.shape}"
    assert torch.all(K_vel_current >= K_vel_min), f"K_vel below minimum: {K_vel_current.min()} < {K_vel_min.min()}"
    assert torch.all(K_vel_current <= K_vel_max), f"K_vel above maximum: {K_vel_current.max()} > {K_vel_max.max()}"


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
@pytest.mark.parametrize("num_envs", [1, 2, 8])
@pytest.mark.parametrize("num_bodies", [1, 4])
def test_lee_pos_randomize_params_within_bounds(
    monkeypatch: pytest.MonkeyPatch, device_str: str, num_envs: int, num_bodies: int
):
    """Randomized gains stay within configured ranges for position controller.

    Tests edge cases with single and multiple environments and bodies.
    """
    device = _device_param(device_str)
    _patch_aggregate(monkeypatch, pos_mod, num_envs, device)
    _patch_sim_context(monkeypatch, pos_mod)
    robot = _DummyRobot(num_envs, num_bodies, device)

    cfg = _create_pos_cfg()
    cfg.randomize_params = True
    controller = pos_mod.LeePosController(cfg, robot, num_envs=num_envs, device=str(device))

    controller.reset_idx(env_ids=None)

    # Check K_pos gains
    K_pos_min = torch.tensor(cfg.K_pos_range[0], device=device, dtype=torch.float32)
    K_pos_max = torch.tensor(cfg.K_pos_range[1], device=device, dtype=torch.float32)
    K_pos_current = controller.K_pos_current.to(device)

    assert K_pos_current.shape == (num_envs, 3), f"Expected shape ({num_envs}, 3), got {K_pos_current.shape}"
    assert torch.all(K_pos_current >= K_pos_min), f"K_pos below minimum: {K_pos_current.min()} < {K_pos_min.min()}"
    assert torch.all(K_pos_current <= K_pos_max), f"K_pos above maximum: {K_pos_current.max()} > {K_pos_max.max()}"


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
@pytest.mark.parametrize("num_envs", [1, 2, 8])
@pytest.mark.parametrize("num_bodies", [1, 4])
def test_lee_acc_randomize_params_within_bounds(
    monkeypatch: pytest.MonkeyPatch, device_str: str, num_envs: int, num_bodies: int
):
    """Randomized gains stay within configured ranges for acceleration controller.

    Tests edge cases with single and multiple environments and bodies.
    """
    device = _device_param(device_str)
    _patch_aggregate(monkeypatch, acc_mod, num_envs, device)
    _patch_sim_context(monkeypatch, acc_mod)
    robot = _DummyRobot(num_envs, num_bodies, device)

    cfg = _create_acc_cfg()
    cfg.randomize_params = True
    controller = acc_mod.LeeAccController(cfg, robot, num_envs=num_envs, device=str(device))

    controller.reset_idx(env_ids=None)

    # Check K_rot gains
    K_rot_min = torch.tensor(cfg.K_rot_range[0], device=device, dtype=torch.float32)
    K_rot_max = torch.tensor(cfg.K_rot_range[1], device=device, dtype=torch.float32)
    K_rot_current = controller.K_rot_current.to(device)

    assert K_rot_current.shape == (num_envs, 3), f"Expected shape ({num_envs}, 3), got {K_rot_current.shape}"
    assert torch.all(K_rot_current >= K_rot_min), f"K_rot below minimum: {K_rot_current.min()} < {K_rot_min.min()}"
    assert torch.all(K_rot_current <= K_rot_max), f"K_rot above maximum: {K_rot_current.max()} > {K_rot_max.max()}"

    # Check K_angvel gains
    K_angvel_min = torch.tensor(cfg.K_angvel_range[0], device=device, dtype=torch.float32)
    K_angvel_max = torch.tensor(cfg.K_angvel_range[1], device=device, dtype=torch.float32)
    K_angvel_current = controller.K_angvel_current.to(device)

    assert K_angvel_current.shape == (num_envs, 3), f"Expected shape ({num_envs}, 3), got {K_angvel_current.shape}"
    assert torch.all(K_angvel_current >= K_angvel_min), (
        f"K_angvel below minimum: {K_angvel_current.min()} < {K_angvel_min.min()}"
    )
    assert torch.all(K_angvel_current <= K_angvel_max), (
        f"K_angvel above maximum: {K_angvel_current.max()} > {K_angvel_max.max()}"
    )


# Cleanup after all tests complete
def teardown_module():
    """Close simulation app after all tests."""
    simulation_app.close()

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
    lee_position_control as pos_mod,
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


def _patch_aggregate(
    monkeypatch: pytest.MonkeyPatch, module, num_envs: int, device: torch.device
) -> None:
    """Monkeypatch aggregate_inertia_about_robot_com to a deterministic CPU-friendly stub."""

    def _agg(*_args, **_kwargs):
        return (
            torch.ones(num_envs, device=device),  # mass
            torch.eye(3, device=device).repeat(num_envs, 1, 1),  # inertia
            torch.zeros((num_envs, 3, 3), device=device),
        )

    monkeypatch.setattr(module.math_utils, "aggregate_inertia_about_robot_com", _agg)


def _device_param(device_str: str) -> torch.device:
    """Return the torch.device or skip when CUDA is unavailable."""
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this system")
    return torch.device(device_str)


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "controller_cls,cfg_cls,mod_name",
    [
        ("LeeVelController", LeeVelControllerCfg, vel_mod),
        ("LeePosController", LeePosControllerCfg, pos_mod),
        ("LeeAccController", LeeAccControllerCfg, acc_mod),
    ],
)
def test_lee_controllers_basic(
    monkeypatch: pytest.MonkeyPatch, device_str: str, controller_cls: str, cfg_cls, mod_name
):
    """Controllers return finite (N, 6) wrench on zero state and counter gravity on +Z."""
    device = _device_param(device_str)
    num_envs, num_bodies = 2, 1
    _patch_aggregate(monkeypatch, mod_name, num_envs, device)
    robot = _DummyRobot(num_envs, num_bodies, device)

    cfg = cfg_cls()
    cfg.randomize_params = False
    controller = getattr(mod_name, controller_cls)(cfg, robot, num_envs=num_envs, device=str(device))

    if controller_cls == "LeeVelController":
        command = torch.zeros((num_envs, 4), device=device)  # vx, vy, vz, yaw_rate
    else:
        command = torch.zeros((num_envs, 3), device=device)  # position or acceleration setpoint

    wrench = controller.compute(command)

    assert wrench.shape == (num_envs, 6)
    assert torch.isfinite(wrench).all()
    assert torch.all(wrench[:, 2] > 0.0)  # body-z force opposes gravity


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_lee_vel_randomize_params_within_bounds(monkeypatch: pytest.MonkeyPatch, device_str: str):
    """Randomized gains stay within configured ranges for velocity controller."""
    device = _device_param(device_str)
    num_envs, num_bodies = 3, 1
    _patch_aggregate(monkeypatch, vel_mod, num_envs, device)
    robot = _DummyRobot(num_envs, num_bodies, device)

    cfg = LeeVelControllerCfg()
    cfg.randomize_params = True
    controller = vel_mod.LeeVelController(cfg, robot, num_envs=num_envs, device=str(device))

    controller.reset_idx(env_ids=None)

    # Ensure tensors are on the correct device
    K_vel_max = torch.tensor(cfg.K_vel_range[0], device=device, dtype=torch.float32)
    K_vel_min = torch.tensor(cfg.K_vel_range[1], device=device, dtype=torch.float32)
        
    # Move controller gains to same device if needed
    K_vel_current = controller.K_vel_current.to(device)

    assert torch.all(K_vel_current >= K_vel_min), f"K_vel below minimum: {K_vel_current} < {K_vel_min}"
    assert torch.all(K_vel_current <= K_vel_max), f"K_vel above maximum: {K_vel_current} > {K_vel_max}"


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_lee_pos_randomize_params_within_bounds(monkeypatch: pytest.MonkeyPatch, device_str: str):
    """Randomized gains stay within configured ranges for position controller."""
    device = _device_param(device_str)
    num_envs, num_bodies = 3, 1
    _patch_aggregate(monkeypatch, pos_mod, num_envs, device)
    robot = _DummyRobot(num_envs, num_bodies, device)

    cfg = LeePosControllerCfg()
    cfg.randomize_params = True
    controller = pos_mod.LeePosController(cfg, robot, num_envs=num_envs, device=str(device))

    controller.reset_idx(env_ids=None)

    # Check K_pos gains
    K_pos_max = torch.tensor(cfg.K_pos_range[0], device=device, dtype=torch.float32)
    K_pos_min = torch.tensor(cfg.K_pos_range[1], device=device, dtype=torch.float32)
    K_pos_current = controller.K_pos_current.to(device)

    assert torch.all(K_pos_current >= K_pos_min), f"K_pos below minimum: {K_pos_current.min()} < {K_pos_min.min()}"
    assert torch.all(K_pos_current <= K_pos_max), f"K_pos above maximum: {K_pos_current.max()} > {K_pos_max.max()}"


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_lee_acc_randomize_params_within_bounds(monkeypatch: pytest.MonkeyPatch, device_str: str):
    """Randomized gains stay within configured ranges for acceleration controller."""
    device = _device_param(device_str)
    num_envs, num_bodies = 3, 1
    _patch_aggregate(monkeypatch, acc_mod, num_envs, device)
    robot = _DummyRobot(num_envs, num_bodies, device)

    cfg = LeeAccControllerCfg()
    cfg.randomize_params = True
    controller = acc_mod.LeeAccController(cfg, robot, num_envs=num_envs, device=str(device))

    controller.reset_idx(env_ids=None)

    # Check K_rot gains
    K_rot_max = torch.tensor(cfg.K_rot_range[0], device=device, dtype=torch.float32)
    K_rot_min = torch.tensor(cfg.K_rot_range[1], device=device, dtype=torch.float32)
    K_rot_current = controller.K_rot_current.to(device)
    
    assert torch.all(K_rot_current >= K_rot_min), f"K_rot below minimum: {K_rot_current.min()} < {K_rot_min.min()}"
    assert torch.all(K_rot_current <= K_rot_max), f"K_rot above maximum: {K_rot_current.max()} > {K_rot_max.max()}"

    # Check K_angvel gains
    K_angvel_max = torch.tensor(cfg.K_angvel_range[0], device=device, dtype=torch.float32)
    K_angvel_min = torch.tensor(cfg.K_angvel_range[1], device=device, dtype=torch.float32)
    K_angvel_current = controller.K_angvel_current.to(device)

    assert torch.all(K_angvel_current >= K_angvel_min), f"K_angvel below minimum: {K_angvel_current.min()} < {K_angvel_min.min()}"
    assert torch.all(K_angvel_current <= K_angvel_max), f"K_angvel above maximum: {K_angvel_current.max()} > {K_angvel_max.max()}"
    

# Cleanup after all tests complete
def teardown_module():
    """Close simulation app after all tests."""
    simulation_app.close()
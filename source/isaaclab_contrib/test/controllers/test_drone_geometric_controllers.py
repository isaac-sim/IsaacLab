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
    lee_attitude_control as att_mod,
)
from isaaclab_contrib.controllers import lee_controller_base as base_mod
from isaaclab_contrib.controllers import (
    lee_position_control as pos_mod,
)
from isaaclab_contrib.controllers import (
    lee_velocity_control as vel_mod,
)
from isaaclab_contrib.controllers.lee_acceleration_control_cfg import LeeAccControllerCfg
from isaaclab_contrib.controllers.lee_attitude_control_cfg import LeeAttControllerCfg
from isaaclab_contrib.controllers.lee_position_control_cfg import LeePosControllerCfg
from isaaclab_contrib.controllers.lee_velocity_control_cfg import LeeVelControllerCfg


class _DummyRobot:
    """Minimal multirotor stub exposing the attributes used by the controllers."""

    def __init__(self, num_envs: int, num_bodies: int, device: torch.device):
        self.num_bodies = num_bodies
        quat_id = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        inertia_flat = torch.eye(3, device=device).reshape(1, 1, 9)
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
            body_mass=torch.ones((num_envs, num_bodies), device=device),
            body_inertia=inertia_flat.repeat(num_envs, num_bodies, 1),
        )


class _DummySimCfg:
    """Mock simulation config."""

    def __init__(self):
        self.gravity = (0.0, 0.0, -9.81)


class _DummySimContext:
    """Mock simulation context."""

    def __init__(self):
        self.cfg = _DummySimCfg()


def _patch_aggregate(monkeypatch, _module, num_envs, device):
    def _agg(*_args, **_kwargs):
        return (
            torch.ones(num_envs, device=device),
            torch.eye(3, device=device).repeat(num_envs, 1, 1),
            torch.zeros((num_envs, 3, 3), device=device),
        )

    monkeypatch.setattr(base_mod, "aggregate_inertia_about_robot_com", _agg)


def _patch_sim_context(monkeypatch: pytest.MonkeyPatch, module) -> None:
    """Monkeypatch SimulationContext.instance() to return a mock."""
    import isaaclab.sim as sim_utils

    def _mock_instance():
        return _DummySimContext()

    monkeypatch.setattr(sim_utils.SimulationContext, "instance", _mock_instance)


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
    cfg.K_vel_range = ((2.5, 2.5, 1.5), (3.5, 3.5, 2.0))
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


def _create_att_cfg() -> LeeAttControllerCfg:
    """Create attitude controller config with required parameters."""
    cfg = LeeAttControllerCfg()
    cfg.K_rot_range = ((1.6, 1.6, 0.25), (1.85, 1.85, 0.4))
    cfg.K_angvel_range = ((0.4, 0.4, 0.075), (0.5, 0.5, 0.09))
    cfg.max_yaw_rate = 1.0471975511965976
    return cfg


# Controllers are pure torch with inertia aggregation patched out: device, env and body counts select no branch.
@pytest.mark.parametrize(
    "controller_cls,cfg_factory,mod_name,gain_names",
    [
        ("LeeVelController", _create_vel_cfg, vel_mod, ("K_vel",)),
        ("LeePosController", _create_pos_cfg, pos_mod, ("K_pos",)),
        ("LeeAccController", _create_acc_cfg, acc_mod, ("K_rot", "K_angvel")),
        ("LeeAttController", _create_att_cfg, att_mod, ("K_rot", "K_angvel")),
    ],
)
def test_lee_controllers_basic(
    monkeypatch: pytest.MonkeyPatch,
    controller_cls: str,
    cfg_factory,
    mod_name,
    gain_names: tuple[str, ...],
):
    """Controllers return finite (N, 6) wrench on zero state, counter gravity on +Z, and randomize gains in range."""
    device = torch.device("cpu")
    num_envs = 2
    _patch_aggregate(monkeypatch, mod_name, num_envs, device)
    _patch_sim_context(monkeypatch, mod_name)
    robot = _DummyRobot(num_envs, 1, device)

    cfg = cfg_factory()
    controller = getattr(mod_name, controller_cls)(cfg, robot, num_envs=num_envs, device=str(device))

    command = torch.zeros((num_envs, 4), device=device)

    wrench = controller.compute(command)

    assert wrench.shape == (num_envs, 6), f"Expected shape ({num_envs}, 6), got {wrench.shape}"
    assert torch.isfinite(wrench).all(), "Wrench contains non-finite values"
    assert torch.all(wrench[:, 2] > 0.0), "Body-z force should oppose gravity"

    controller.reset_idx(env_ids=None)

    for name in gain_names:
        gain_range = getattr(cfg, f"{name}_range")
        gain_min = torch.tensor(gain_range[0], device=device, dtype=torch.float32)
        gain_max = torch.tensor(gain_range[1], device=device, dtype=torch.float32)
        gain = getattr(controller, f"{name}_current").to(device)

        assert gain.shape == (num_envs, 3), f"Expected {name} shape ({num_envs}, 3), got {gain.shape}"
        assert torch.all(gain >= gain_min), f"{name} below minimum: {gain.min()} < {gain_min.min()}"
        assert torch.all(gain <= gain_max), f"{name} above maximum: {gain.max()} > {gain_max.max()}"

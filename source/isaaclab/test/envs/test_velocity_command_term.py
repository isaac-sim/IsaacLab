# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the uniform velocity command term."""

import math
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit


@contextmanager
def _raise_on_cuda_sync(device: str):
    """Raise if the enclosed code synchronizes a CUDA device."""
    if not device.startswith("cuda"):
        yield
        return
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def _make_command(device: str, num_envs: int) -> tuple[UniformVelocityCommand, SimpleNamespace]:
    """Create a heading-controlled velocity command on a stub robot."""
    robot_data = SimpleNamespace(
        heading_w=SimpleNamespace(torch=torch.zeros(num_envs, device=device)),
        root_lin_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3, device=device)),
        root_ang_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3, device=device)),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device=device,
        scene={"robot": SimpleNamespace(data=robot_data)},
        sim=MagicMock(),
        extras={},
    )
    cfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=True,
        heading_control_stiffness=0.8,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
    return UniformVelocityCommand(cfg, env), robot_data


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_heading_control_updates_only_heading_envs(device):
    """Heading envs track the wrapped, clipped heading error without a sync; others keep their sampled yaw rate."""
    heading_target = [3.0, 0.5, -3.0, 0.2, 1.0, -0.1]
    heading = [-3.0, 0.0, 3.0, 0.0, -1.0, 0.0]
    is_heading_env = [True, True, True, False, True, False]
    sampled_yaw_rate = [0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    command, robot_data = _make_command(device, len(heading))
    command.heading_target[:] = torch.tensor(heading_target, device=device)
    robot_data.heading_w.torch[:] = torch.tensor(heading, device=device)
    command.is_heading_env[:] = torch.tensor(is_heading_env, device=device)
    command.is_standing_env[:] = False
    command.vel_command_b[:] = 0.25
    command.vel_command_b[:, 2] = torch.tensor(sampled_yaw_rate, device=device)

    with _raise_on_cuda_sync(device):
        command._update_command()

    expected_yaw_rate = []
    for target, current, active, sampled in zip(heading_target, heading, is_heading_env, sampled_yaw_rate):
        error = math.atan2(math.sin(target - current), math.cos(target - current))
        expected_yaw_rate.append(min(max(0.8 * error, -1.0), 1.0) if active else sampled)
    torch.testing.assert_close(command.command[:, 2].cpu(), torch.tensor(expected_yaw_rate))
    torch.testing.assert_close(command.command[:, :2].cpu(), torch.full((len(heading), 2), 0.25))


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_reset_logs_metrics_without_host_reads(device):
    """Reset logs episode metrics as device scalars without synchronizing the stream."""
    command, robot_data = _make_command(device, 4)
    robot_data.root_lin_vel_b.torch[:2, 0] = 1.0
    command.compute(dt=0.1)

    with _raise_on_cuda_sync(device):
        extras = command.reset()

    success_rate = command._env.extras["log"]["Metrics/success_rate"]
    for value in (*extras.values(), success_rate):
        assert isinstance(value, torch.Tensor) and value.ndim == 0
    assert set(extras) == {"error_vel_xy", "error_vel_yaw"}
    torch.testing.assert_close(success_rate.cpu(), torch.tensor(0.5))

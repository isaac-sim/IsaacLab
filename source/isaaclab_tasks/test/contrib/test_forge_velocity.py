# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for FORGE's finite-difference velocity observations."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import warp as wp

from isaaclab_tasks.contrib.forge.forge_env import ForgeEnv

pytestmark = pytest.mark.unit


def test_velocity_uses_consecutive_noisy_poses():
    """Preserve translation, yaw and noise across the real parent/child updates."""
    env = object.__new__(ForgeEnv)
    env.sim = SimpleNamespace(device="cpu")
    env.scene = SimpleNamespace(num_envs=1, env_origins=torch.zeros((1, 3)))
    env.cfg = SimpleNamespace(
        obs_rand=SimpleNamespace(fingertip_pos=0.00025, fingertip_rot_deg=math.degrees(0.002), ft_force=0.0),
        ft_smoothing_factor=0.25,
    )
    zeros = torch.zeros((1, 3))
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Upright tool: roll pi, XYZW.
    asset_data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=zeros.clone()), root_quat_w=SimpleNamespace(torch=quat.clone())
    )
    env._fixed_asset = SimpleNamespace(data=asset_data)
    env._held_asset = SimpleNamespace(data=asset_data)
    positions = torch.zeros((1, 3, 3))
    env._robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=positions),
            body_quat_w=SimpleNamespace(torch=quat[:, None].repeat(1, 3, 1)),
            body_lin_vel_w=SimpleNamespace(torch=torch.zeros((1, 3, 3))),
            body_ang_vel_w=SimpleNamespace(torch=torch.zeros((1, 3, 3))),
            body_link_jacobian_w=SimpleNamespace(torch=torch.zeros((1, 3, 6, 7))),
            mass_matrix=SimpleNamespace(torch=torch.eye(7)[None]),
            joint_pos=SimpleNamespace(torch=torch.zeros((1, 9))),
            joint_vel=SimpleNamespace(torch=torch.zeros((1, 9))),
        ),
        _data=SimpleNamespace(_sim_timestamp=0.0),
        root_view=SimpleNamespace(
            get_link_incoming_joint_force=lambda: wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu")
        ),
    )
    env.fingertip_body_idx, env.left_finger_body_idx, env.right_finger_body_idx = 0, 1, 2
    env.force_sensor_body_idx = 0
    env.prev_fingertip_pos = zeros.clone()
    env.prev_fingertip_quat = quat.clone()
    env.prev_joint_pos = torch.zeros((1, 7))
    env.flip_quats = torch.ones(1)
    env.force_sensor_world_smooth = torch.zeros((1, 6))
    env.fixed_pos_obs_frame = zeros.clone()
    env.init_fixed_pos_obs_noise = zeros.clone()

    # Translate 1 mm and rotate 0.01 rad per 10 ms, then hold still.
    # Opposite noise in the last two samples must contribute both previous and current errors.
    samples = (
        (0.001, 0.01, 0.0, 0.0, 0.1, 1.0),
        (0.002, 0.02, 0.0, 0.0, 0.1, 1.0),
        (0.003, 0.03, 0.00025, 1.0, 0.125, 1.2),
        (0.003, 0.03, -0.00025, -1.0, -0.05, -0.4),
    )
    for x, yaw, pos_noise, rot_noise, expected_vx, expected_wz in samples:
        positions[0, 0, 0] = x
        env._robot.data.body_quat_w.torch[:] = torch.tensor([math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0])
        # Supply deterministic position noise, rotation axis/angle and force noise.
        draws = [
            torch.tensor([[pos_noise / env.cfg.obs_rand.fingertip_pos, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, -1.0]]),  # Local -Z is world +Z for the upright tool.
            torch.tensor([rot_noise]),
            zeros.clone(),
        ]
        with patch("torch.randn", side_effect=draws):
            env._compute_intermediate_values(dt=0.01)
        torch.testing.assert_close(env.ee_linvel_fd, torch.tensor([[expected_vx, 0.0, 0.0]]))
        torch.testing.assert_close(env.ee_angvel_fd, torch.tensor([[0.0, 0.0, expected_wz]]), rtol=1e-4, atol=1e-6)

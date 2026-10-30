# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for FORGE's finite-difference velocity observations."""

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab_tasks.contrib.factory.factory_env import FactoryEnv
from isaaclab_tasks.contrib.forge.forge_env import ForgeEnv

pytestmark = pytest.mark.unit


@pytest.fixture
def forge_env():
    """Provide backend pose samples while executing real tensor initialization and updates."""
    env = object.__new__(ForgeEnv)
    env.sim = SimpleNamespace(device="cpu")
    env.scene = SimpleNamespace(num_envs=1, env_origins=torch.zeros((1, 3)))
    env.cfg = SimpleNamespace(
        obs_rand=SimpleNamespace(fingertip_pos=0.0, fingertip_rot_deg=0.0, ft_force=0.0),
        ctrl=SimpleNamespace(
            ema_factor=0.1,
            ema_factor_range=(0.025, 0.1),
            pos_action_bounds=[0.05] * 3,
            task_prop_gains_noise_level=[0.0] * 6,
            pos_threshold_noise_level=[0.0] * 3,
            rot_threshold_noise_level=[0.0] * 3,
        ),
        task=SimpleNamespace(contact_penalty_threshold_range=(1.0, 2.0)),
        ft_smoothing_factor=0.25,
    )
    zeros = torch.zeros((1, 3))
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Upright tool: roll pi, XYZW.
    asset_data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=zeros.clone()), root_quat_w=SimpleNamespace(torch=quat.clone())
    )
    env._fixed_asset = SimpleNamespace(data=asset_data)
    env._held_asset = SimpleNamespace(data=asset_data)
    env._robot = SimpleNamespace(
        body_names=["panda_fingertip_centered", "panda_leftfinger", "panda_rightfinger"],
        num_joints=9,
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=torch.zeros((1, 3, 3))),
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
    env._init_tensors()
    env.force_sensor_body_idx = 0
    env.flip_quats = torch.ones(1)
    env.force_sensor_world_smooth = torch.zeros((1, 6))
    env.default_gains = torch.ones((1, 6))
    env.default_pos_threshold = torch.ones((1, 3))
    env.default_rot_threshold = torch.ones((1, 3))
    env.default_dead_zone = torch.zeros((1, 6))
    env.actions = torch.zeros((1, 7))
    env.prev_actions = env.actions.clone()
    return env


@pytest.mark.parametrize("add_noise", [False, True])
def test_velocity_uses_consecutive_noisy_poses(forge_env, add_noise):
    """Compare velocity with independently recorded poses, allowing sampling to be reordered."""
    env = forge_env
    if add_noise:
        env.cfg.obs_rand.fingertip_pos = 0.00025
        env.cfg.obs_rand.fingertip_rot_deg = math.degrees(0.002)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        env._compute_intermediate_values(dt=0.01)
        previous_pos = env.noisy_fingertip_pos.clone()
        previous_yaw = 2 * torch.atan2(env.noisy_fingertip_quat[:, 1], env.noisy_fingertip_quat[:, 0])
        for step in range(1, 5):
            env._robot.data.body_pos_w.torch[0, 0, 0] = step * 0.001
            yaw = step * 0.01
            env._robot.data.body_quat_w.torch[:] = torch.tensor([math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0])
            env._compute_intermediate_values(dt=0.01)
            expected_linear = (env.noisy_fingertip_pos - previous_pos) / 0.01
            current_yaw = 2 * torch.atan2(env.noisy_fingertip_quat[:, 1], env.noisy_fingertip_quat[:, 0])
            delta_yaw = current_yaw - previous_yaw
            expected_yaw_rate = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw)) / 0.01
            torch.testing.assert_close(env.ee_linvel_fd, expected_linear)
            torch.testing.assert_close(env.ee_angvel_fd[:, 2], expected_yaw_rate, rtol=1e-4, atol=1e-5)
            if not add_noise:
                torch.testing.assert_close(env.ee_linvel_fd, torch.tensor([[0.1, 0.0, 0.0]]))
                torch.testing.assert_close(env.ee_angvel_fd, torch.tensor([[0.0, 0.0, 1.0]]), rtol=1e-4, atol=1e-5)
            previous_pos = env.noisy_fingertip_pos.clone()
            previous_yaw = current_yaw.clone()


def test_reset_rebases_noisy_pose_history(forge_env, monkeypatch):
    """A reset pose jump must not become a policy velocity in the next stationary sample."""
    env = forge_env
    env._compute_intermediate_values(dt=0.01)

    def reset_backend(self, env_ids):
        # Supply the parent's completed reset pose/observation without running physics settling.
        self._robot.data.body_pos_w.torch[:, 0] = torch.tensor([[1.0, -0.5, 0.2]])
        yaw = 0.8
        self._robot.data.body_quat_w.torch[:] = torch.tensor([math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0])
        self.fingertip_midpoint_pos = self._robot.data.body_pos_w.torch[:, 0].clone()
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w.torch[:, 0].clone()
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.noisy_fingertip_quat = self.fingertip_midpoint_quat.clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()
        self.ee_linvel_fd.zero_()
        self.ee_angvel_fd.zero_()

    monkeypatch.setattr(FactoryEnv, "_reset_idx", reset_backend)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        env._reset_idx(torch.tensor([0]))
        env._compute_intermediate_values(dt=0.01)
    torch.testing.assert_close(env.ee_linvel_fd, torch.zeros((1, 3)))
    torch.testing.assert_close(env.ee_angvel_fd, torch.zeros((1, 3)), rtol=0.0, atol=1e-5)

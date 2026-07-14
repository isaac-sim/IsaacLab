# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from isaaclab_tasks.core.locomotion.locomotion_direct_env import LocomotionDirectEnv

_NUM_ENVS = 4
_NUM_JOINTS = 3


def _make_environment() -> tuple[LocomotionDirectEnv, Mock]:
    env = object.__new__(LocomotionDirectEnv)
    env._is_closed = True
    reset_time_outs = torch.tensor([False, True, False, True])
    default_joint_pos = torch.arange(_NUM_ENVS * _NUM_JOINTS, dtype=torch.float32).reshape(_NUM_ENVS, _NUM_JOINTS)
    default_joint_vel = default_joint_pos + 20.0
    default_root_pose = torch.zeros((_NUM_ENVS, 7), dtype=torch.float32)
    default_root_pose[:, 6] = 1.0
    default_root_vel = torch.arange(_NUM_ENVS * 6, dtype=torch.float32).reshape(_NUM_ENVS, 6)

    robot = Mock()
    robot.data = SimpleNamespace(
        default_joint_pos=SimpleNamespace(torch=default_joint_pos),
        default_joint_vel=SimpleNamespace(torch=default_joint_vel),
        default_root_pose=SimpleNamespace(torch=default_root_pose),
        default_root_vel=SimpleNamespace(torch=default_root_vel),
    )
    scene = SimpleNamespace(
        num_envs=_NUM_ENVS,
        env_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]]),
    )
    scene.reset = Mock(side_effect=robot.reset)

    env.robot = robot
    env.scene = scene
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1.0 / 120.0),
        events=None,
        action_noise_model=None,
        observation_noise_model=None,
    )
    env.reset_time_outs = reset_time_outs
    env.extras = {}
    env.episode_length_buf = torch.arange(_NUM_ENVS)
    env.targets = torch.tensor([1000.0, 0.0, 0.0]).repeat(_NUM_ENVS, 1)
    env.potentials = torch.zeros(_NUM_ENVS)
    env._compute_intermediate_values = Mock()
    return env, robot


def test_reset_idx_resets_articulation_once_through_scene():
    env, robot = _make_environment()
    env_ids = torch.tensor([1, 3])

    env._reset_idx(env_ids)

    env.scene.reset.assert_called_once_with(env_ids)
    robot.reset.assert_called_once_with(env_ids)


def test_reset_idx_keeps_success_rate_on_device():
    env, _ = _make_environment()
    env_ids = torch.tensor([0, 1, 2])

    env._reset_idx(env_ids)

    success_rate = env.extras["log"]["Metrics/success_rate"]
    assert isinstance(success_rate, torch.Tensor)
    assert success_rate.device == env.reset_time_outs.device
    torch.testing.assert_close(success_rate, torch.tensor(1.0 / 3.0))


def test_reset_idx_uses_fused_joint_state_writer():
    env, robot = _make_environment()
    env_ids = torch.tensor([1, 3])

    env._reset_idx(env_ids)

    robot.write_joint_state_to_sim_index.assert_called_once()
    call = robot.write_joint_state_to_sim_index.call_args
    torch.testing.assert_close(call.kwargs["position"], robot.data.default_joint_pos.torch[env_ids])
    torch.testing.assert_close(call.kwargs["velocity"], robot.data.default_joint_vel.torch[env_ids])
    torch.testing.assert_close(call.kwargs["env_ids"], env_ids)
    robot.write_joint_position_to_sim_index.assert_not_called()
    robot.write_joint_velocity_to_sim_index.assert_not_called()


def test_reset_idx_does_not_mutate_default_state():
    env, robot = _make_environment()
    env_ids = torch.tensor([1, 3])
    default_root_pose = robot.data.default_root_pose.torch.clone()
    default_root_velocity = robot.data.default_root_vel.torch.clone()
    default_joint_position = robot.data.default_joint_pos.torch.clone()
    default_joint_velocity = robot.data.default_joint_vel.torch.clone()

    env._reset_idx(env_ids)

    torch.testing.assert_close(robot.data.default_root_pose.torch, default_root_pose)
    torch.testing.assert_close(robot.data.default_root_vel.torch, default_root_velocity)
    torch.testing.assert_close(robot.data.default_joint_pos.torch, default_joint_position)
    torch.testing.assert_close(robot.data.default_joint_vel.torch, default_joint_velocity)

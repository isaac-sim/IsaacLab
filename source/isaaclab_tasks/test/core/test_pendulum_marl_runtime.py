# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Runtime tests for the multi-agent pendulum task."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture(scope="module")
def pendulum_env():
    sim_utils.create_new_stage()
    cfg = parse_env_cfg("Isaac-Pendulum-MARL-Direct", device="cuda", num_envs=2)
    cfg.seed = 42
    env = gym.make("Isaac-Pendulum-MARL-Direct", cfg=cfg)
    try:
        yield env
    finally:
        env.close()
        sim_utils.close_stage()


def test_pendulum_marl_runtime_interface(pendulum_env):
    env = pendulum_env
    observations, _ = env.reset()
    actions = {agent: torch.zeros((2, 1), device=env.unwrapped.device) for agent in env.unwrapped.possible_agents}

    observations, rewards, terminated, truncated, _ = env.step(actions)

    assert env.unwrapped.possible_agents == ["cart", "pendulum"]
    assert observations["cart"].shape == (2, 4)
    assert observations["pendulum"].shape == (2, 3)
    assert rewards.keys() == {"cart", "pendulum"}
    assert terminated.keys() == {"cart", "pendulum"}
    assert truncated.keys() == {"cart", "pendulum"}
    assert torch.equal(terminated["cart"], terminated["pendulum"])
    assert torch.equal(truncated["cart"], truncated["pendulum"])
    assert env.unwrapped.state().shape == (2, 7)


def test_pendulum_success_lifecycle(pendulum_env):
    env = pendulum_env
    env.reset()
    raw_env = env.unwrapped
    required_steps = raw_env._success_required_steps
    raw_env._consecutive_upright_steps.fill_(required_steps - 1)

    joint_pos = raw_env.robot.data.joint_pos.torch.clone()
    joint_pos[:, raw_env._pole_dof_idx] = 0.0
    joint_pos[:, raw_env._pendulum_dof_idx] = 0.0
    raw_env.robot.write_joint_position_to_sim_index(position=joint_pos)
    raw_env._get_dones()
    assert torch.equal(
        raw_env._consecutive_upright_steps,
        torch.full((2,), required_steps, dtype=torch.long, device=raw_env.device),
    )

    joint_pos[1, raw_env._pendulum_dof_idx] = 2.0 * raw_env.cfg.success_upright_angle
    raw_env.robot.write_joint_position_to_sim_index(position=joint_pos)
    raw_env._get_dones()
    assert raw_env._consecutive_upright_steps[0].item() == required_steps + 1
    assert raw_env._consecutive_upright_steps[1].item() == 0

    env_ids = torch.arange(2, device=raw_env.device, dtype=torch.long)
    raw_env._consecutive_upright_steps[env_ids] = torch.tensor(
        [required_steps, required_steps - 1], device=raw_env.device
    )
    done_false = torch.zeros(2, dtype=torch.bool, device=raw_env.device)
    time_out_true = torch.ones(2, dtype=torch.bool, device=raw_env.device)
    raw_env.time_out_dict = {agent: time_out_true for agent in raw_env.possible_agents}
    raw_env.terminated_dict = {agent: done_false for agent in raw_env.possible_agents}

    raw_env._reset_idx(env_ids)

    assert raw_env.extras["log"]["Metrics/success_rate"] == pytest.approx(0.5)
    assert torch.equal(
        raw_env._consecutive_upright_steps[env_ids],
        torch.zeros(2, dtype=torch.long, device=raw_env.device),
    )

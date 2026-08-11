# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the multi-agent pendulum task."""

import importlib.util
import math

import gymnasium as gym
import pytest
import torch

from isaaclab.envs import DirectMARLEnv

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.pendulum.pendulum_marl_env import (
    PendulumMARLEnv,
    compute_rewards,
    compute_success,
    links_upright,
    normalize_angle,
    update_upright_steps,
)
from isaaclab_tasks.core.pendulum.pendulum_marl_env_cfg import PendulumMARLEnvCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def test_only_explicit_marl_task_is_registered():
    spec = gym.spec("Isaac-Pendulum-MARL-Direct")
    cfg = load_cfg_from_registry("Isaac-Pendulum-MARL-Direct", "env_cfg_entry_point")

    assert spec.entry_point.endswith("pendulum_marl_env:PendulumMARLEnv")
    assert isinstance(cfg, PendulumMARLEnvCfg)
    assert issubclass(PendulumMARLEnv, DirectMARLEnv)
    with pytest.raises(gym.error.NameNotFound):
        gym.spec("Isaac-Pendulum-Direct")
    with pytest.raises(gym.error.NameNotFound):
        gym.spec("Isaac-Pendulum")


def test_marl_registration_uses_explicit_agent_config_names():
    spec = gym.spec("Isaac-Pendulum-MARL-Direct")
    assert spec.kwargs["rl_games_cfg_entry_point"].endswith("rl_games_marl_ppo_cfg.yaml")
    assert spec.kwargs["skrl_cfg_entry_point"].endswith("skrl_marl_ppo_cfg.yaml")
    assert spec.kwargs["skrl_ippo_cfg_entry_point"].endswith("skrl_marl_ippo_cfg.yaml")
    assert spec.kwargs["skrl_mappo_cfg_entry_point"].endswith("skrl_marl_mappo_cfg.yaml")


@pytest.mark.parametrize(
    "module_name",
    [
        "isaaclab_tasks.core.pendulum.pendulum_direct_env",
        "isaaclab_tasks.core.pendulum.pendulum_direct_env_cfg",
        "isaaclab_tasks.core.pendulum.pendulum_manager_env_cfg",
        "isaaclab_tasks.core.pendulum.mdp",
    ],
)
def test_single_agent_modules_are_removed(module_name: str):
    assert importlib.util.find_spec(module_name) is None


def test_marl_rewards_preserve_the_existing_formulas_without_step_dt():
    rewards = compute_rewards(
        1.0,
        -2.0,
        -0.01,
        -1.0,
        -0.01,
        -1.0,
        -0.01,
        torch.tensor([0.4]),
        torch.tensor([0.2]),
        torch.tensor([-0.5]),
        torch.tensor([-0.3]),
        torch.tensor([0.6]),
        torch.tensor([False]),
    )
    torch.testing.assert_close(rewards["cart"], torch.tensor([0.951]))
    torch.testing.assert_close(rewards["pendulum"], torch.tensor([0.984]))


def test_links_upright_uses_physical_link_angles_and_inclusive_boundary():
    boundary_angle = math.pi / 12
    max_angle = normalize_angle(torch.tensor(boundary_angle)).item()
    epsilon = 1.0e-4
    pole_pos = torch.tensor([boundary_angle, -boundary_angle, 0.2, 0.2, boundary_angle + epsilon, 0.0, 0.0, 0.0])
    pendulum_pos = torch.tensor(
        [
            -boundary_angle,
            boundary_angle,
            -0.4,
            0.2,
            -(boundary_angle + epsilon),
            boundary_angle + epsilon,
            boundary_angle,
            -boundary_angle,
        ]
    )

    upright = links_upright(pole_pos, pendulum_pos, max_angle)

    assert torch.equal(upright, torch.tensor([True, True, True, False, False, False, True, True]))


def test_upright_counter_requires_consecutive_steps():
    steps = torch.tensor([59, 12, 60])
    upright = torch.tensor([True, False, True])

    updated = update_upright_steps(steps, upright)

    assert torch.equal(updated, torch.tensor([60, 0, 61]))


def test_success_requires_timeout_no_termination_and_full_window():
    success = compute_success(
        time_out=torch.tensor([True, True, True, False]),
        terminated=torch.tensor([False, False, True, False]),
        upright_steps=torch.tensor([60, 59, 60, 100]),
        required_steps=60,
    )

    assert torch.equal(success, torch.tensor([True, False, False, False]))


@pytest.mark.parametrize(
    "agent_cfg_entry_point",
    [
        "rl_games_cfg_entry_point",
        "skrl_cfg_entry_point",
        "skrl_ippo_cfg_entry_point",
        "skrl_mappo_cfg_entry_point",
    ],
)
def test_marl_agent_registry_configs_load_as_dict(agent_cfg_entry_point: str):
    cfg = load_cfg_from_registry("Isaac-Pendulum-MARL-Direct", agent_cfg_entry_point)

    assert isinstance(cfg, dict)


def test_marl_config_preserves_agents_spaces_and_timing():
    cfg = PendulumMARLEnvCfg()
    assert cfg.possible_agents == ["cart", "pendulum"]
    assert cfg.action_spaces == {"cart": 1, "pendulum": 1}
    assert cfg.observation_spaces == {"cart": 4, "pendulum": 3}
    assert cfg.state_space == -1
    assert cfg.scene.num_envs == 4096
    assert cfg.scene.env_spacing == 4.0
    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.decimation == 2
    assert cfg.episode_length_s == 5.0
    assert round(cfg.success_duration_s / (cfg.sim.dt * cfg.decimation)) == 60

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract tests for the manager-based Pendulum MARL task."""

from types import SimpleNamespace

import gymnasium as gym
import torch

import isaaclab_tasks.core.pendulum  # noqa: F401
from isaaclab_tasks.core.pendulum import mdp
from isaaclab_tasks.core.pendulum.pendulum_marl_manager_env_cfg import PendulumMARLManagerEnvCfg


def test_manager_task_registration_uses_manager_marl_env() -> None:
    """Register the manager task against the manager-based MARL environment."""
    spec = gym.spec("Isaac-Pendulum-MARL")

    assert spec.entry_point == "isaaclab.envs:ManagerBasedMARLEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(":PendulumMARLManagerEnvCfg")
    assert spec.kwargs["skrl_ippo_cfg_entry_point"].endswith("skrl_marl_ippo_cfg.yaml")
    assert spec.kwargs["skrl_mappo_cfg_entry_point"].endswith("skrl_marl_mappo_cfg.yaml")


def test_manager_config_preserves_agents_spaces_timing_and_state() -> None:
    """Keep the final direct-task agent order, manager spaces, timing, and state order."""
    cfg = PendulumMARLManagerEnvCfg()

    assert list(cfg.agents) == ["cart", "pendulum"]
    assert cfg.decimation == 2
    assert cfg.episode_length_s == 5.0
    assert cfg.sim.dt == 1 / 120
    assert cfg.sim.render_interval == 2
    assert cfg.scene.num_envs == 4096
    assert cfg.scene.env_spacing == 4.0
    assert cfg.agents["cart"].actions.effort.joint_names == ["slider_to_cart"]
    assert cfg.agents["cart"].actions.effort.scale == 100.0
    assert cfg.agents["pendulum"].actions.effort.joint_names == ["pole_to_pendulum"]
    assert cfg.agents["pendulum"].actions.effort.scale == 50.0
    assert cfg.agents["cart"].observations.policy.observations.func is mdp.cart_observation
    assert cfg.agents["pendulum"].observations.policy.observations.func is mdp.pendulum_observation
    assert cfg.state.state.observations.func is mdp.state


def test_reward_terms_produce_expected_final_7025_values() -> None:
    """Keep the shared per-step reward numerically identical to the final direct task."""
    reward_cfg = PendulumMARLManagerEnvCfg().agents["cart"].rewards
    assert [
        reward_cfg.alive.weight,
        reward_cfg.terminating.weight,
        reward_cfg.cart_vel.weight,
        reward_cfg.pole_pos.weight,
        reward_cfg.pole_vel.weight,
        reward_cfg.pendulum_pos.weight,
        reward_cfg.pendulum_vel.weight,
        reward_cfg.upright.weight,
        reward_cfg.action.weight,
    ] == [1.0, -2.0, -0.01, 1.0, -0.01, 1.0, -0.01, 1.0, -0.01]

    step_dt = 1 / 60
    reward = mdp.compute_rewards(
        1.0,
        -2.0,
        -0.01,
        1.0,
        -0.01,
        1.0,
        -0.01,
        1.0,
        -0.01,
        torch.tensor([0.25, -0.5]),
        torch.tensor([0.0, torch.pi]),
        torch.tensor([0.5, -0.25]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.25, 0.5]),
        torch.tensor([True, False]),
        torch.tensor([[0.5], [1.0]]),
        torch.tensor([[0.5], [-0.5]]),
        torch.tensor([False, True]),
        step_dt,
    )

    expected = torch.tensor([3.98, -4.0225]) * step_dt
    assert torch.allclose(reward, expected)


def test_success_tracker_updates_once_and_resets_selected_envs() -> None:
    """Update once per control step and retain tracker state for unselected environments."""
    parent = SimpleNamespace(extras={"log": {}})
    agent = SimpleNamespace(
        parent=parent,
        termination_manager=SimpleNamespace(
            time_outs=torch.tensor([True, True]),
            terminated=torch.tensor([False, True]),
        ),
    )
    tracker = object.__new__(mdp.ConsecutiveUprightSuccess)
    tracker._env = agent
    tracker._upright_steps = torch.zeros(2, dtype=torch.long)
    tracker._success_required_steps = 2

    tracker.update(torch.tensor([True, False]))
    tracker.update(torch.tensor([True, False]))
    assert torch.equal(tracker._upright_steps, torch.tensor([2, 0]))

    tracker._upright_steps[1] = 5
    tracker.reset(torch.tensor([0]))
    assert parent.extras["log"]["Metrics/success_rate"] == 1.0
    assert torch.equal(tracker._upright_steps, torch.tensor([0, 5]))

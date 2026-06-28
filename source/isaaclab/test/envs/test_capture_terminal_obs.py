# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace

import torch

from isaaclab.envs import DirectMARLEnv, DirectRLEnv


class _AddNoise:
    """Deterministic stand-in for a noise model: adds a fixed offset to the observation."""

    def __init__(self, offset: float):
        self.offset = offset

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data + self.offset


def test_direct_capture_terminal_obs_without_noise():
    """The terminal observation is the pre-reset observation, unmodified when no noise is configured."""
    env = DirectRLEnv.__new__(DirectRLEnv)
    policy = torch.tensor([[1.0], [2.0]])
    critic = torch.tensor([[5.0], [6.0]])
    env._get_observations = lambda: {"policy": policy.clone(), "critic": critic.clone()}
    env.cfg = SimpleNamespace(observation_noise_model=None)

    terminal_obs = env._capture_terminal_obs()

    assert torch.equal(terminal_obs["policy"], policy)
    assert torch.equal(terminal_obs["critic"], critic)


def test_direct_capture_terminal_obs_applies_noise_to_policy_only():
    """Noise is applied to the policy observation but not the state/critic space."""
    env = DirectRLEnv.__new__(DirectRLEnv)
    policy = torch.tensor([[1.0], [2.0]])
    critic = torch.tensor([[5.0], [6.0]])
    env._get_observations = lambda: {"policy": policy.clone(), "critic": critic.clone()}
    # a truthy cfg flag selects the noise path; the model itself is the callable below
    env.cfg = SimpleNamespace(observation_noise_model=object())
    env._observation_noise_model = _AddNoise(offset=10.0)

    terminal_obs = env._capture_terminal_obs()

    assert torch.equal(terminal_obs["policy"], policy + 10.0)
    assert torch.equal(terminal_obs["critic"], critic)


def test_marl_capture_terminal_obs_applies_noise_per_agent():
    """Per-agent noise is applied only to agents that have a noise model configured."""
    env = DirectMARLEnv.__new__(DirectMARLEnv)
    obs_0 = torch.tensor([[1.0], [2.0]])
    obs_1 = torch.tensor([[3.0], [4.0]])
    env._get_observations = lambda: {"agent_0": obs_0.clone(), "agent_1": obs_1.clone()}
    env.cfg = SimpleNamespace(observation_noise_model={"agent_0": object()})
    env._observation_noise_model = {"agent_0": _AddNoise(offset=100.0)}

    terminal_obs = env._capture_terminal_obs()

    # agent_0 has a noise model -> noised; agent_1 has none -> untouched
    assert torch.equal(terminal_obs["agent_0"], obs_0 + 100.0)
    assert torch.equal(terminal_obs["agent_1"], obs_1)

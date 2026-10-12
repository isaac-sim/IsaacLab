# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for direct single-agent reinforcement-learning environments."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils.noise import (
    ConstantNoiseCfg,
    GaussianNoiseCfg,
    NoiseModelWithAdditiveBias,
    NoiseModelWithAdditiveBiasCfg,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "enable_noise, policy_shape, sample_bias_per_component",
    [
        (False, (2, 3), False),
        (True, (2, 3), False),
        (True, (2, 3, 4, 5), False),
        (True, (2, 3, 4, 5), True),
        (True, (2,), False),
        (True, (2,), True),
    ],
)
def test_reset_observations(enable_noise: bool, policy_shape: tuple[int, ...], sample_bias_per_component: bool) -> None:
    """Reset returns noisy policy observations after resampling bias, while leaving critic observations clean."""
    noise_cfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=ConstantNoiseCfg(bias=0.5),
        bias_noise_cfg=GaussianNoiseCfg(mean=1.0, std=0.0, operation="abs"),
        sample_bias_per_component=sample_bias_per_component,
    )
    env = object.__new__(DirectRLEnv)
    env._is_closed = True
    env.cfg = DirectRLEnvCfg(observation_noise_model=noise_cfg if enable_noise else None)
    env.scene = SimpleNamespace(num_envs=2, reset=lambda ids: None, write_data_to_sim=lambda: None)
    env.sim = SimpleNamespace(
        device="cpu", forward=lambda: None, render_context=SimpleNamespace(reset_scene_state_cadence=lambda: None)
    )
    env.has_rtx_sensors = False
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    env.extras = {}
    env._get_observations = lambda: {"policy": torch.zeros(policy_shape), "critic": torch.full((2, 4), 4.0)}
    if enable_noise:
        env._observation_noise_model = NoiseModelWithAdditiveBias(noise_cfg, num_envs=2, device="cpu")

    for bias in (1.0, 2.0):
        noise_cfg.bias_noise_cfg.mean = bias
        observations, _ = env.reset()

        expected = bias + 0.5 if enable_noise else 0.0
        torch.testing.assert_close(observations["policy"], torch.full(policy_shape, expected))
        torch.testing.assert_close(observations["critic"], torch.full((2, 4), 4.0))

    if enable_noise:
        noise_cfg.bias_noise_cfg.std = 1.0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            observations, _ = env.reset()
        policy = observations["policy"]
        assert policy.shape == policy_shape
        assert not torch.equal(policy[0], policy[1])
        if sample_bias_per_component and policy.ndim > 1:
            assert not torch.equal(policy[:, 0], policy[:, 1])
        elif not sample_bias_per_component:
            flat_policy = policy.reshape(2, -1)
            torch.testing.assert_close(flat_policy, flat_policy[:, :1].expand_as(flat_policy))
        torch.testing.assert_close(env._compute_observations()["policy"], policy)

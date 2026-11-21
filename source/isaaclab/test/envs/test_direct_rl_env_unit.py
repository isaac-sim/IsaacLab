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


@pytest.mark.parametrize("multi_agent", [False, True])
def test_every_step_final_observations_own_nested_tensor_leaves(multi_agent: bool) -> None:
    """Composite terminal observations retain pre-reset values after live buffers change."""
    from isaaclab.envs import DirectMARLEnv

    live = torch.ones(2, 1)
    observations = {"policy": {"position": (live,)}}
    done, timeout = torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool)
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            action_noise_model=None,
            decimation=1,
            sim=SimpleNamespace(render_interval=1),
            final_obs_every_step=True,
            compute_final_obs=True,
            events=None,
            possible_agents=["policy"],
        ),
        extras={"policy": {}, "time_outs": torch.zeros(2, dtype=torch.bool)} if multi_agent else {},
        device="cpu",
        num_envs=2,
        physics_dt=0.01,
        step_dt=0.01,
        sim=SimpleNamespace(is_rendering=False, step=lambda **kwargs: None, consume_reset_request=lambda: False),
        scene=SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **kwargs: None),
        _physics_handles_decimation=False,
        _sim_step_counter=0,
        common_step_counter=0,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        reset_buf=torch.zeros(2, dtype=torch.bool),
        reset_terminated=torch.zeros(2, dtype=torch.bool),
        reset_time_outs=timeout.clone(),
        _pre_physics_step=lambda actions: None,
        _apply_action=lambda: None,
        _compute_observations=lambda: observations,
        _get_rewards=lambda: {"policy": torch.zeros(2)} if multi_agent else torch.zeros(2),
        _get_dones=lambda: ({"policy": done}, {"policy": timeout}) if multi_agent else (done, timeout),
        _reset_idx=lambda ids: live.index_fill_(0, ids.long(), -100),
        render_enabled=False,
        video_recorders=[],
        possible_agents=["policy"],
    )
    env._reset_envs_from_buffer = lambda: DirectRLEnv._reset_envs_from_buffer(env)
    if multi_agent:
        DirectMARLEnv.step(env, {"policy": torch.zeros(2, 1)})
        final = env.extras["policy"]["final_obs"]
    else:
        DirectRLEnv.step(env, torch.zeros(2, 1))
        final = env.extras["final_obs"]["policy"]
    torch.testing.assert_close(final["position"][0], torch.ones(2, 1))
    assert live[0].item() == -100 and live[1].item() == 1


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

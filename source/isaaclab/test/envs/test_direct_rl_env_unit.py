# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for direct single-agent reinforcement-learning environments."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ManagerBasedRLEnv
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


@pytest.mark.parametrize("env_type", [DirectRLEnv, ManagerBasedRLEnv])
@pytest.mark.kitless
def test_ui_resets_rerender_before_observations_and_recording(env_type):
    """A UI reset refreshes cameras even when another environment already autoreset in the same step."""
    env = object.__new__(env_type)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=1,
        sim=SimpleNamespace(render_interval=100, dt=0.01),
        compute_final_obs=False,
        action_noise_model=None,
        observation_noise_model=None,
        num_rerenders_on_reset=2,
        events=False,
    )
    state = torch.zeros(2, 1)
    image = torch.zeros_like(state)
    done = torch.tensor([True, False])
    timeout = torch.zeros_like(done)
    render_calls = []

    def reset(ids):
        state[ids] = 1

    def render(**kwargs):
        render_calls.append(True)
        image.copy_(state)

    def post_reset(ids):
        torch.testing.assert_close(image[ids], state[ids])

    env.sim = SimpleNamespace(
        device="cpu", is_rendering=True, consume_reset_request=lambda: True, step=lambda **kwargs: None, render=render
    )
    env.scene = SimpleNamespace(num_envs=2, write_data_to_sim=lambda: None, update=lambda **kwargs: None)
    env.render_enabled = env.has_rtx_sensors = True
    env.video_recorders = []
    env._physics_handles_actions = env._physics_handles_decimation = False
    env._sim_step_counter = env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.reset_buf = torch.zeros_like(done)
    env.reset_terminated = torch.zeros_like(done)
    env.reset_time_outs = torch.zeros_like(done)
    env._reset_idx = reset
    env.extras = {}
    if env_type is ManagerBasedRLEnv:
        env.action_manager = SimpleNamespace(process_action=lambda action: None, apply_action=lambda: None)
        env.termination_manager = SimpleNamespace(compute=lambda: done.clone(), terminated=done, time_outs=timeout)
        env.reward_manager = SimpleNamespace(compute=lambda **kwargs: torch.zeros(2))
        env.command_manager = SimpleNamespace(compute=lambda **kwargs: None)
        env.event_manager = SimpleNamespace(active_terms=[])
        env.observation_manager = SimpleNamespace(compute=lambda **kwargs: {"policy": image.clone()})
        env.recorder_manager = SimpleNamespace(
            record_pre_step=lambda: None,
            record_post_physics_decimation_step=lambda: None,
            record_post_step=lambda: None,
            record_pre_reset=lambda ids: None,
            record_post_reset=post_reset,
        )
    else:
        env._pre_physics_step = lambda actions: None
        env._apply_action = lambda: None
        env._get_dones = lambda: (done, timeout)
        env._get_rewards = lambda: torch.zeros(2)
        env._get_observations = lambda: {"policy": image.clone()}

    observations, *_ = env.step(torch.zeros(2, 1))
    torch.testing.assert_close(observations["policy"], torch.ones_like(image))
    assert len(render_calls) == 4

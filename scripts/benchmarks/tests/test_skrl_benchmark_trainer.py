# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for BenchmarkTrainer — run with a fake env and fake agent.

These tests do NOT spin up Isaac Sim. They verify the trainer's
per-iteration capture logic in isolation.
"""

from __future__ import annotations

import time

import pytest
import torch

from scripts.benchmarks.skrl_benchmark_trainer import BenchmarkTrainer


class _FakeEnv:
    """Minimal env compatible with SKRL's SequentialTrainer expectations."""

    num_agents = 1
    num_envs = 4
    state_space = None
    observation_space = type("O", (), {"shape": (2,)})()
    action_space = type("A", (), {"shape": (1,)})()
    device = torch.device("cpu")

    def __init__(self, reward_schedule):
        self._rewards = reward_schedule  # list[float] — one per step
        self._i = 0

    def reset(self):
        return torch.zeros(self.num_envs, 2), {}

    def step(self, actions):
        r = self._rewards[self._i % len(self._rewards)]
        self._i += 1
        rewards = torch.full((self.num_envs,), float(r))
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        next_states = torch.zeros(self.num_envs, 2)
        return next_states, rewards, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


class _FakeAgent:
    """Minimal agent that exposes `_rollouts`, pre/post_interaction, track_data."""

    def __init__(self, rollouts: int = 4):
        self._rollouts = rollouts
        self.tracking_data: dict[str, list[float]] = {}
        self._init_called = False
        self._running_mode = None

    def init(self, trainer_cfg):
        self._init_called = True

    def set_running_mode(self, mode):
        self._running_mode = mode

    def pre_interaction(self, timestep, timesteps):
        pass

    def act(self, states, timestep, timesteps):
        return torch.zeros(states.shape[0], 1), None, None

    def record_transition(self, **kwargs):
        pass

    def post_interaction(self, timestep, timesteps):
        pass

    def track_data(self, tag, value):
        self.tracking_data.setdefault(tag, []).append(value)


def test_iter_times_s_length_matches_iterations():
    rollouts = 4
    max_iters = 3
    env = _FakeEnv(reward_schedule=[1.0] * 100)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}

    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    assert len(trainer.iter_times_s) == max_iters
    assert all(t > 0.0 for t in trainer.iter_times_s)


def test_iter_rewards_reflects_synthetic_schedule():
    rollouts = 4
    max_iters = 3
    # Give each rollout a distinguishable reward value.
    schedule = [1.0] * rollouts + [2.0] * rollouts + [3.0] * rollouts
    env = _FakeEnv(reward_schedule=schedule)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}

    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    # Each iteration's mean reward = mean over rollouts*num_envs rewards.
    # For constant-per-rollout schedules: iter k ≈ schedule[k*rollouts].
    assert trainer.iter_rewards == pytest.approx([1.0, 2.0, 3.0])


def test_iter_ep_lengths_defaults_to_zero_when_no_termination():
    rollouts = 4
    max_iters = 2
    env = _FakeEnv(reward_schedule=[0.0] * 100)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}

    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    # Fake env never terminates → ep_lengths fall back to 0.0 each iter.
    assert trainer.iter_ep_lengths == [0.0, 0.0]


def test_iter_times_s_shows_variance_with_sleep():
    """Real per-iter timing must vary when iterations take different wall times."""
    rollouts = 2
    max_iters = 2

    class _SlowEnv(_FakeEnv):
        def step(self, actions):
            if self._i == 0 or self._i == 1:
                time.sleep(0.02)
            return super().step(actions)

    env = _SlowEnv(reward_schedule=[0.0] * 100)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}

    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    assert len(trainer.iter_times_s) == max_iters
    # First iter had two sleep(0.02) calls (steps 0 and 1); second iter didn't.
    # Accept any positive separation; this is about existence of variance, not magnitude.
    assert trainer.iter_times_s[0] > trainer.iter_times_s[1]


def test_multi_env_does_not_call_env_reset_on_termination():
    """Regression: Task 4's initial fix unconditionally reset on any termination,
    which corrupts multi-env VecEnv training (parent's single_agent_train guards
    this on num_envs > 1)."""
    rollouts = 4
    max_iters = 2

    class _CountingMultiEnv(_FakeEnv):
        num_envs = 8  # multi-env — parent must NOT mid-train reset

        def __init__(self, reward_schedule):
            super().__init__(reward_schedule=reward_schedule)
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1
            return torch.zeros(self.num_envs, 2), {}

        def step(self, actions):
            r = self._rewards[self._i % len(self._rewards)]
            self._i += 1
            rewards = torch.full((self.num_envs,), float(r))
            # Terminate env 0 on every step — should NOT trigger env.reset()
            terminated = torch.zeros(self.num_envs, dtype=torch.bool)
            terminated[0] = True
            truncated = torch.zeros(self.num_envs, dtype=torch.bool)
            next_states = torch.zeros(self.num_envs, 2)
            return next_states, rewards, terminated, truncated, {}

    env = _CountingMultiEnv(reward_schedule=[1.0] * 100)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}
    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    # Exactly one reset — the initial one at loop start.
    assert env.reset_calls == 1, (
        f"BenchmarkTrainer called env.reset() {env.reset_calls} times on a "
        f"multi-env VecEnv. Parent single_agent_train only resets at start "
        f"when num_envs > 1 — VecEnv handles per-env auto-reset internally."
    )


def test_single_env_resets_when_episode_ends():
    """Sanity: the single-env branch still resets on termination."""
    rollouts = 2
    max_iters = 1

    class _CountingSingleEnv(_FakeEnv):
        num_envs = 1

        def __init__(self, reward_schedule):
            super().__init__(reward_schedule=reward_schedule)
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1
            return torch.zeros(self.num_envs, 2), {}

        def step(self, actions):
            r = self._rewards[self._i % len(self._rewards)]
            self._i += 1
            rewards = torch.full((self.num_envs,), float(r))
            terminated = torch.zeros(self.num_envs, dtype=torch.bool)
            terminated[0] = True  # terminate every step on num_envs=1
            truncated = torch.zeros(self.num_envs, dtype=torch.bool)
            next_states = torch.zeros(self.num_envs, 2)
            return next_states, rewards, terminated, truncated, {}

    env = _CountingSingleEnv(reward_schedule=[0.0] * 100)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}
    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    # Initial reset (1) + per-step reset on each termination (rollouts=2)
    # = 3 total.
    assert env.reset_calls >= 2, f"Expected ≥2 resets on single-env terminations, got {env.reset_calls}"

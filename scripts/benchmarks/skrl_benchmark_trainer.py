# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BenchmarkTrainer — SKRL trainer subclass that captures per-iteration metrics.

Mirrors :class:`skrl.trainers.torch.SequentialTrainer`'s training loop and
records, once per rollout-buffer fill (= one iteration):

* ``iter_times_s``  - wall-clock seconds from the first env step of the
  rollout to just after ``agent.post_interaction`` of the rollout's final
  step (i.e. after the PPO update).
* ``iter_rewards``  - mean reward across all env steps and all parallel
  envs during the rollout.
* ``iter_ep_lengths`` - last value of
  ``agent.tracking_data["Episode / Total timesteps (mean)"]`` observed at
  iteration end, or ``0.0`` when no episode terminated yet.

These attributes are populated after :meth:`train` returns and are read
directly by ``benchmark_skrl.py``'s v1 bundle builder — no TB round trip.
"""

from __future__ import annotations

import inspect
import time

import torch
import tqdm
from skrl.trainers.torch import SequentialTrainer

# skrl >= ~2.x removed the ``agents_scope`` keyword from
# ``SequentialTrainer.__init__``. Detect once at import time so the wrapper
# stays compatible with both old and new versions without try/except per
# call site.
_SUPER_INIT_PARAMS = inspect.signature(SequentialTrainer.__init__).parameters
_SUPPORTS_AGENTS_SCOPE = "agents_scope" in _SUPER_INIT_PARAMS


class BenchmarkTrainer(SequentialTrainer):
    """SequentialTrainer that records per-iteration timing + reward + ep length."""

    def __init__(self, env, agents, agents_scope=None, cfg=None) -> None:
        if _SUPPORTS_AGENTS_SCOPE:
            super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=cfg)
        else:
            super().__init__(env=env, agents=agents, cfg=cfg)
        self.iter_times_s: list[float] = []
        self.iter_rewards: list[float] = []
        self.iter_ep_lengths: list[float] = []

    def train(self) -> None:
        # Exactly one non-simultaneous single-agent training path — mirrors
        # the parent SequentialTrainer for that case. If the user is running
        # multi-agent or simultaneous agents, defer to the stock loop (those
        # paths don't populate the per-iteration benchmark attributes).
        if self.num_simultaneous_agents > 1 or self.env.num_agents > 1:
            super().train()
            return

        rollouts_attr = getattr(self.agents, "_rollouts", None)
        if not rollouts_attr:
            # Agent has no rollout boundary (e.g. off-policy SAC/DDPG).
            # Defer to the stock training loop — the per-iter attributes
            # stay empty, and benchmark_skrl.py will treat that as "no
            # per-iter data available" rather than wall-time garbage.
            super().train()
            return
        rollouts = int(rollouts_attr)
        max_iters = self.timesteps // rollouts

        self.agents.set_running_mode("train")
        states, infos = self.env.reset()

        iter_start_ns = time.perf_counter_ns()
        rollout_reward_sum = 0.0
        rollout_reward_count = 0

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                if not self.headless:
                    self.env.render()

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

                rollout_reward_sum += float(rewards.mean().item())
                rollout_reward_count += 1

            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # Reset envs only when running a single env; multi-env VecEnvs
            # handle per-env resets themselves. Mirrors
            # skrl.trainers.torch.base.Trainer.single_agent_train.
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

            # One iteration = one rollout-buffer fill.
            if (timestep + 1) % rollouts == 0:
                iter_end_ns = time.perf_counter_ns()
                self.iter_times_s.append((iter_end_ns - iter_start_ns) / 1e9)
                mean_reward = rollout_reward_sum / max(rollout_reward_count, 1)
                self.iter_rewards.append(mean_reward)
                ep_len_samples = self.agents.tracking_data.get("Episode / Total timesteps (mean)", [])
                self.iter_ep_lengths.append(float(ep_len_samples[-1]) if ep_len_samples else 0.0)
                # Reset per-iter accumulators + timer for the next rollout.
                iter_start_ns = time.perf_counter_ns()
                rollout_reward_sum = 0.0
                rollout_reward_count = 0

        # Cap any series to max_iters (guards against off-by-one if timesteps
        # isn't a clean multiple of rollouts).
        self.iter_times_s = self.iter_times_s[:max_iters]
        self.iter_rewards = self.iter_rewards[:max_iters]
        self.iter_ep_lengths = self.iter_ep_lengths[:max_iters]

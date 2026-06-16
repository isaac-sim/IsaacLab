# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-backend descriptors for the unified training benchmark.

Pure data — no RL library imports. The thin per-backend adapters under
``scripts/benchmarks/<backend>/`` consume these to know each framework's
TensorBoard event-file location and which scalar tags carry the reward /
episode-length series. Framework-specific *launch* logic stays in the adapters;
only the declarative metadata lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.test.benchmark.schema import Framework


@dataclass(frozen=True)
class BackendDescriptor:
    """Declarative metadata for one RL backend's benchmark integration.

    Args:
        framework: Schema framework id.
        tfevents_pattern: Glob (relative to the run ``log_dir``) matching the
            TensorBoard events file — ``"events*"`` (root), ``"summaries/events*"``
            (rl_games), or ``"PPO_*/events*"`` (sb3).
        reward_tag: TensorBoard scalar tag for mean reward per iteration.
        ep_length_tag: TensorBoard scalar tag for mean episode length per iteration.
    """

    framework: Framework
    tfevents_pattern: str
    reward_tag: str
    ep_length_tag: str


_DESCRIPTORS = [
    BackendDescriptor(
        framework="rsl_rl",
        tfevents_pattern="events*",
        reward_tag="Train/mean_reward",
        ep_length_tag="Train/mean_episode_length",
    ),
    BackendDescriptor(
        framework="rl_games",
        tfevents_pattern="summaries/events*",
        reward_tag="rewards/iter",
        ep_length_tag="episode_lengths/iter",
    ),
    BackendDescriptor(
        framework="skrl",
        tfevents_pattern="events*",
        reward_tag="Reward / Total reward (mean)",
        ep_length_tag="Episode / Total timesteps (mean)",
    ),
    BackendDescriptor(
        framework="sb3",
        tfevents_pattern="PPO_*/events*",
        reward_tag="rollout/ep_rew_mean",
        ep_length_tag="rollout/ep_len_mean",
    ),
]

BACKEND_DESCRIPTORS: dict[Framework, BackendDescriptor] = {d.framework: d for d in _DESCRIPTORS}

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per rl library descriptors to unify the field names in the training benchmark.

Thin rl-library adapters to map library specific names to a unified naming convention.
It maps TensorBoard event-file location and which scalar tags carry the reward / episode-length series.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.test.benchmark.schema import Framework


@dataclass(frozen=True)
class RLLibraryDescriptor:
    """Declarative metadata for one RL library benchmark integration.

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


BACKEND_DESCRIPTORS: dict[Framework, RLLibraryDescriptor] = {
    "rsl_rl": BackendDescriptor(
        framework="rsl_rl",
        tfevents_pattern="events*",
        reward_tag="Train/mean_reward",
        ep_length_tag="Train/mean_episode_length",
    ),
    "rl_games": BackendDescriptor(
        framework="rl_games",
        tfevents_pattern="summaries/events*",
        reward_tag="rewards/iter",
        ep_length_tag="episode_lengths/iter",
    ),
    "skrl": BackendDescriptor(
        framework="skrl",
        tfevents_pattern="events*",
        reward_tag="Reward / Total reward (mean)",
        ep_length_tag="Episode / Total timesteps (mean)",
    ),
    "sb3": BackendDescriptor(
        framework="sb3",
        tfevents_pattern="PPO_*/events*",
        reward_tag="rollout/ep_rew_mean",
        ep_length_tag="rollout/ep_len_mean",
    ),
}

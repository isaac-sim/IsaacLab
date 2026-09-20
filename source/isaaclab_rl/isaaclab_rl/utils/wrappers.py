# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared validation for reinforcement learning environment wrappers."""

import gymnasium as gym


def _validate_no_time_limit(env: gym.Env) -> None:
    """Reject Gymnasium time limits around Isaac Lab vectorized environments.

    Isaac Lab tracks episode timeouts independently for every sub-environment. Gymnasium's
    :class:`~gymnasium.wrappers.TimeLimit` instead tracks a scalar number of calls to ``step``
    and returns a scalar truncation flag, which violates the vectorized environment contract.

    Args:
        env: The environment and its Gymnasium wrapper chain.

    Raises:
        ValueError: If the wrapper chain contains a Gymnasium time limit.
    """
    current_env = env
    while isinstance(current_env, gym.Wrapper):
        if isinstance(current_env, gym.wrappers.TimeLimit):
            raise ValueError(
                "Gymnasium's TimeLimit wrapper is incompatible with Isaac Lab vectorized environments. "
                "Configure env.cfg.episode_length_s for per-environment timeouts and create the environment "
                "without max_episode_steps."
            )
        current_env = current_env.env

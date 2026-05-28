# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Random-action sampler shared across the benchmark scripts.

Single-agent (``DirectRLEnv`` / ``ManagerBasedRLEnv``) envs expose
``single_action_space``; multi-agent (``DirectMARLEnv``) envs expose
``action_spaces`` — a dict keyed by agent id. ``env.step`` accepts the
matching shape: a stacked tensor for single-agent, a dict of stacked
tensors for multi-agent. The benchmark startup phase needs random
actions for the first env step and previously assumed single-agent;
this helper picks the right shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["sample_random_actions"]


def sample_random_actions(env: Any) -> torch.Tensor | dict[str, torch.Tensor]:
    """Sample one random action per env from the env's action space(s).

    Discriminates single-agent from multi-agent by duck typing on
    ``action_spaces`` (plural, dict-valued). DirectRLEnv and
    ManagerBasedRLEnv expose ``single_action_space``; DirectMARLEnv
    exposes ``action_spaces``. Both shapes ultimately get fed straight
    to ``env.step``.

    Args:
        env: The benchmark target — typically a ``gym.Env`` returned by
            ``gym.make``. The unwrapped env must expose ``num_envs``
            and ``device`` plus either ``single_action_space`` or
            ``action_spaces``.

    Returns:
        A ``torch.Tensor`` of shape ``(num_envs, action_dim)`` for
        single-agent envs, or a dict ``{agent: tensor}`` for
        multi-agent envs. dtype is ``torch.float32`` on the env's
        device.
    """
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "action_spaces"):
        return {
            agent: torch.as_tensor(
                np.stack([space.sample() for _ in range(unwrapped.num_envs)]),
                dtype=torch.float32,
                device=unwrapped.device,
            )
            for agent, space in unwrapped.action_spaces.items()
        }
    np_actions = np.stack([unwrapped.single_action_space.sample() for _ in range(unwrapped.num_envs)])
    return torch.as_tensor(np_actions, dtype=torch.float32, device=unwrapped.device)

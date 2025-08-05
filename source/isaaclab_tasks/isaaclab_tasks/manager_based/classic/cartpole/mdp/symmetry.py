# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to specify the symmetry in the observation and action space for cartpole."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    two symmetrical transformations: original, left-right. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor. Defaults to None.
        actions: The original actions tensor. Defaults to None.
        obs_type: The type of observation to augment. Defaults to "policy".

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        num_envs = obs.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = torch.zeros(num_envs * 2, obs.shape[1], device=obs.device)
        # -- original
        obs_aug[:num_envs] = obs[:]
        # -- left-right
        obs_aug[num_envs : 2 * num_envs] = -obs
    else:
        obs_aug = None

    # actions
    if actions is not None:
        num_envs = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(num_envs * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:num_envs] = actions[:]
        # -- left-right
        actions_aug[num_envs : 2 * num_envs] = -actions
    else:
        actions_aug = None

    return obs_aug, actions_aug

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        obs_aug = obs.repeat(4)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
        # -- front-back
        obs_aug["policy"][2 * batch_size : 3 * batch_size] = _transform_policy_obs_front_back(
            env.unwrapped, obs["policy"]
        )
        # -- diagonal
        obs_aug["policy"][3 * batch_size :] = _transform_policy_obs_front_back(
            env.unwrapped, obs_aug["policy"][batch_size : 2 * batch_size]
        )
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(batch_size * 4, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
        # -- front-back
        actions_aug[2 * batch_size : 3 * batch_size] = _transform_actions_front_back(actions)
        # -- diagonal
        actions_aug[3 * batch_size :] = _transform_actions_front_back(actions_aug[batch_size : 2 * batch_size])
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    perm, sign = _policy_obs_symmetry("left_right", obs.shape[1], _has_height_scan(env), str(obs.device))
    return obs[:, perm] * sign


def _transform_policy_obs_front_back(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes negating
    certain components of the linear and angular velocities, projected gravity, velocity commands,
    and flipping the joint positions, joint velocities, and last actions for the ANYmal robot.
    Additionally, if height-scan data is present, it is flipped along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with front-back symmetry applied.
    """
    perm, sign = _policy_obs_symmetry("front_back", obs.shape[1], _has_height_scan(env), str(obs.device))
    return obs[:, perm] * sign


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    perm, sign = _joint_symmetry("left_right", str(actions.device))
    return actions[:, perm] * sign


def _transform_actions_front_back(actions: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with front-back symmetry applied.
    """
    perm, sign = _joint_symmetry("front_back", str(actions.device))
    return actions[:, perm] * sign


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
    'LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA',
    'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE',
    'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE'
]

Correspondingly, the joint ordering for the ANYmal robot is:

* LF = left front --> [0, 4, 8]
* LH = left hind --> [1, 5, 9]
* RF = right front --> [2, 6, 10]
* RH = right hind --> [3, 7, 11]

Each transform is a column permutation and a sign per column: ``out[:, i] = sign[i] * x[:, perm[i]]``.
"""

# signs of the base lin vel, ang vel, projected gravity, and velocity command columns
_BASE_SIGNS = {
    "left_right": (1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1),
    "front_back": (-1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1),
}
# source joint of each switched joint: left <-> right swaps LF/RF and LH/RH, front <-> back swaps LF/LH and RF/RH
_JOINT_PERMS = {
    "left_right": (2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9),
    "front_back": (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10),
}
# left <-> right flips the HAA joints, front <-> back flips the HFE and KFE joints
_JOINT_SIGNS = {
    "left_right": (-1,) * 4 + (1,) * 8,
    "front_back": (1,) * 4 + (-1,) * 8,
}
# height-scan grid dimension flipped per transform, for the (11, 17) grid
_HEIGHT_SCAN_FLIP_DIM = {"left_right": 0, "front_back": 1}


def _has_height_scan(env: ManagerBasedRLEnv) -> bool:
    """Whether the policy observation contains the height scan."""
    return "height_scan" in env.observation_manager.active_terms["policy"]


@functools.cache
def _joint_symmetry(kind: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Column permutation and signs of a transform over the 12 joints, cached per device."""
    return (
        torch.tensor(_JOINT_PERMS[kind], dtype=torch.long, device=device),
        torch.tensor(_JOINT_SIGNS[kind], device=device),
    )


@functools.cache
def _policy_obs_symmetry(kind: str, num_obs: int, height_scan: bool, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Column permutation and signs of a transform over the policy observation, cached per device."""
    perm = list(range(num_obs))
    sign = [1] * num_obs
    sign[:12] = _BASE_SIGNS[kind]
    # joint positions, joint velocities, and last actions
    for start in (12, 24, 36):
        perm[start : start + 12] = [start + joint for joint in _JOINT_PERMS[kind]]
        sign[start : start + 12] = _JOINT_SIGNS[kind]
    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if height_scan:
        grid = torch.arange(48, 235).view(11, 17).flip(dims=[_HEIGHT_SCAN_FLIP_DIM[kind]])
        perm[48:235] = grid.flatten().tolist()
    return (
        torch.tensor(perm, dtype=torch.long, device=device),
        torch.tensor(sign, device=device),
    )

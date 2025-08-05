# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal."""

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
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
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
        # since we have 4 different symmetries, we need to augment the batch size by 4
        obs_aug = torch.zeros(num_envs * 4, obs.shape[1], device=obs.device)
        # -- original
        obs_aug[:num_envs] = obs[:]
        # -- left-right
        obs_aug[num_envs : 2 * num_envs] = _transform_obs_left_right(env.unwrapped, obs, obs_type)
        # -- front-back
        obs_aug[2 * num_envs : 3 * num_envs] = _transform_obs_front_back(env.unwrapped, obs, obs_type)
        # -- diagonal
        obs_aug[3 * num_envs :] = _transform_obs_front_back(env.unwrapped, obs_aug[num_envs : 2 * num_envs])
    else:
        obs_aug = None

    # actions
    if actions is not None:
        num_envs = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(num_envs * 4, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:num_envs] = actions[:]
        # -- left-right
        actions_aug[num_envs : 2 * num_envs] = _transform_actions_left_right(actions)
        # -- front-back
        actions_aug[2 * num_envs : 3 * num_envs] = _transform_actions_front_back(actions)
        # -- diagonal
        actions_aug[3 * num_envs :] = _transform_actions_front_back(actions_aug[num_envs : 2 * num_envs])
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor, obs_type: str = "policy") -> torch.Tensor:
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
        obs_type: The type of observation to augment. Defaults to "policy".

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, :3] = obs[:, :3] * torch.tensor([1, -1, 1], device=device)
    # ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 12:24] = _switch_anymal_joints_left_right(obs[:, 12:24])
    # joint vel
    obs[:, 24:36] = _switch_anymal_joints_left_right(obs[:, 24:36])
    # last actions
    obs[:, 36:48] = _switch_anymal_joints_left_right(obs[:, 36:48])

    # height-scan
    if obs_type == "critic":
        # handle asymmetric actor-critic formulation
        group_name = "critic" if "critic" in env.observation_manager.active_terms else "policy"
    else:
        group_name = "policy"

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms[group_name]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)

    return obs


def _transform_obs_front_back(env: ManagerBasedRLEnv, obs: torch.Tensor, obs_type: str = "policy") -> torch.Tensor:
    """Applies a front-back symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes negating
    certain components of the linear and angular velocities, projected gravity, velocity commands,
    and flipping the joint positions, joint velocities, and last actions for the ANYmal robot.
    Additionally, if height-scan data is present, it is flipped along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.
        obs_type: The type of observation to augment. Defaults to "policy".

    Returns:
        The transformed observation tensor with front-back symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, :3] = obs[:, :3] * torch.tensor([-1, 1, 1], device=device)
    # ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([-1, 1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([-1, 1, -1], device=device)
    # joint pos
    obs[:, 12:24] = _switch_anymal_joints_front_back(obs[:, 12:24])
    # joint vel
    obs[:, 24:36] = _switch_anymal_joints_front_back(obs[:, 24:36])
    # last actions
    obs[:, 36:48] = _switch_anymal_joints_front_back(obs[:, 36:48])

    # height-scan
    if obs_type == "critic":
        # handle asymmetric actor-critic formulation
        group_name = "critic" if "critic" in env.observation_manager.active_terms else "policy"
    else:
        group_name = "policy"

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms[group_name]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[2]).view(-1, 11 * 17)

    return obs


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
    actions = actions.clone()
    actions[:] = _switch_anymal_joints_left_right(actions[:])
    return actions


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
    actions = actions.clone()
    actions[:] = _switch_anymal_joints_front_back(actions[:])
    return actions


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
"""


def _switch_anymal_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    joint_data_switched[..., [0, 4, 8, 1, 5, 9]] = joint_data[..., [2, 6, 10, 3, 7, 11]]
    # right <-- left
    joint_data_switched[..., [2, 6, 10, 3, 7, 11]] = joint_data[..., [0, 4, 8, 1, 5, 9]]

    # Flip the sign of the HAA joints
    joint_data_switched[..., [0, 1, 2, 3]] *= -1.0

    return joint_data_switched


def _switch_anymal_joints_front_back(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # front <-- hind
    joint_data_switched[..., [0, 4, 8, 2, 6, 10]] = joint_data[..., [1, 5, 9, 3, 7, 11]]
    # hind <-- front
    joint_data_switched[..., [1, 5, 9, 3, 7, 11]] = joint_data[..., [0, 4, 8, 2, 6, 10]]

    # Flip the sign of the HFE and KFE joints
    joint_data_switched[..., 4:] *= -1

    return joint_data_switched

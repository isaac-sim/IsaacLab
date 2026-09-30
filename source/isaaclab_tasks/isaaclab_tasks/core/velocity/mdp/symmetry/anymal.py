# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

import functools
import re
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

    # resolve the joint permutations from the robot's joint names (cached per joint order)
    joint_names = env.unwrapped.scene["robot"].joint_names
    device = obs["policy"].device if obs is not None else actions.device
    left_right, front_back = _anymal_joint_symmetry_maps(tuple(joint_names), str(device))

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        obs_aug = obs.repeat(4)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"], left_right
        )
        # -- front-back
        obs_aug["policy"][2 * batch_size : 3 * batch_size] = _transform_policy_obs_front_back(
            env.unwrapped, obs["policy"], front_back
        )
        # -- diagonal
        obs_aug["policy"][3 * batch_size :] = _transform_policy_obs_front_back(
            env.unwrapped, obs_aug["policy"][batch_size : 2 * batch_size], front_back
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
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions, left_right)
        # -- front-back
        actions_aug[2 * batch_size : 3 * batch_size] = _transform_actions_front_back(actions, front_back)
        # -- diagonal
        actions_aug[3 * batch_size :] = _transform_actions_front_back(
            actions_aug[batch_size : 2 * batch_size], front_back
        )
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(
    env: ManagerBasedRLEnv, obs: torch.Tensor, joint_map: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
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
        joint_map: The left-right joint permutation and sign in the robot's joint order.

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
    obs[:, 12:24] = _switch_joints(obs[:, 12:24], joint_map)
    # joint vel
    obs[:, 24:36] = _switch_joints(obs[:, 24:36], joint_map)
    # last actions
    obs[:, 36:48] = _switch_joints(obs[:, 36:48], joint_map)

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)

    return obs


def _transform_policy_obs_front_back(
    env: ManagerBasedRLEnv, obs: torch.Tensor, joint_map: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes negating
    certain components of the linear and angular velocities, projected gravity, velocity commands,
    and flipping the joint positions, joint velocities, and last actions for the ANYmal robot.
    Additionally, if height-scan data is present, it is flipped along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.
        joint_map: The front-back joint permutation and sign in the robot's joint order.

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
    obs[:, 12:24] = _switch_joints(obs[:, 12:24], joint_map)
    # joint vel
    obs[:, 24:36] = _switch_joints(obs[:, 24:36], joint_map)
    # last actions
    obs[:, 36:48] = _switch_joints(obs[:, 36:48], joint_map)

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[2]).view(-1, 11 * 17)

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor, joint_map: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.
        joint_map: The left-right joint permutation and sign in the robot's joint order.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    return _switch_joints(actions, joint_map)


def _transform_actions_front_back(actions: torch.Tensor, joint_map: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.
        joint_map: The front-back joint permutation and sign in the robot's joint order.

    Returns:
        The transformed actions tensor with front-back symmetry applied.
    """
    return _switch_joints(actions, joint_map)


"""
Helper functions for symmetry.

The ANYmal joints are named ``<side><end>_<joint>``, where the side is ``L`` (left) or ``R`` (right), the end is
``F`` (front) or ``H`` (hind), and the joint is ``HAA`` (hip abduction/adduction), ``HFE`` (hip flexion/extension)
or ``KFE`` (knee flexion/extension). The joint order of the articulation depends on the physics backend and on
:attr:`~isaaclab.assets.ArticulationCfg.joint_ordering`, so the mirrored joints are resolved by name:

* left-right: swap ``L`` and ``R`` and negate the ``HAA`` joints.
* front-back: swap ``F`` and ``H`` and negate the ``HFE`` and ``KFE`` joints.
"""

_ANYMAL_JOINT_NAME = re.compile(r"(?P<side>[LR])(?P<end>[FH])_(?P<joint>HAA|HFE|KFE)")


@functools.cache
def _anymal_joint_symmetry_maps(
    joint_names: tuple[str, ...], device: str
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Resolves the left-right and front-back joint permutations and signs from the joint names.

    Args:
        joint_names: The joint names of the robot, in the order of the joint observations and actions.
        device: The device on which to create the tensors.

    Returns:
        The ``(indices, signs)`` pairs for the left-right and the front-back symmetry. The mirrored value of
        joint ``i`` is ``signs[i] * data[..., indices[i]]``.

    Raises:
        ValueError: If a joint name does not follow the ANYmal naming convention or has no mirrored counterpart.
    """
    parsed = []
    for name in joint_names:
        match = _ANYMAL_JOINT_NAME.fullmatch(name)
        if match is None:
            raise ValueError(
                f"Joint '{name}' does not follow the ANYmal naming convention: {_ANYMAL_JOINT_NAME.pattern}."
            )
        parsed.append(match.groupdict())
    name_to_index = {name: i for i, name in enumerate(joint_names)}

    def _resolve(swap: dict[str, str], negated: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        indices = []
        for p in parsed:
            counterpart = f"{swap.get(p['side'], p['side'])}{swap.get(p['end'], p['end'])}_{p['joint']}"
            if counterpart not in name_to_index:
                raise ValueError(f"Mirrored joint '{counterpart}' is missing from the joint names: {joint_names}.")
            indices.append(name_to_index[counterpart])
        signs = [-1.0 if p["joint"] in negated else 1.0 for p in parsed]
        return torch.tensor(indices, device=device), torch.tensor(signs, device=device)

    left_right = _resolve({"L": "R", "R": "L"}, negated=("HAA",))
    front_back = _resolve({"F": "H", "H": "F"}, negated=("HFE", "KFE"))
    return left_right, front_back


def _switch_joints(joint_data: torch.Tensor, joint_map: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Applies a joint permutation and sign flip to the last dimension of the joint data tensor."""
    indices, signs = joint_map
    return joint_data[..., indices] * signs

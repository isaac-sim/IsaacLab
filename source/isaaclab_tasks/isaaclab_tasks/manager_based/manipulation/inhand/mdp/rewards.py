# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def success_bonus(
    env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Bonus reward for successfully reaching the goal.

    The object is considered to have reached the goal when the object orientation is within the threshold.
    The reward is 1.0 if the object has reached the goal, otherwise 0.0.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # obtain the threshold for the orientation error
    threshold = command_term.cfg.orientation_success_threshold
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)

    return dtheta <= threshold


def track_pos_l2(
    env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward for tracking the object position using the L2 norm.

    The reward is the distance between the object position and the goal position.

    Args:
        env: The environment object.
        command_term: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal position
    goal_pos_e = command_term.command[:, 0:3]
    # obtain the object position in the environment frame
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins

    return torch.norm(goal_pos_e - object_pos_e, p=2, dim=-1)


def track_orientation_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 1e-3,
) -> torch.Tensor:
    """Reward for tracking the object orientation using the inverse of the orientation error.

    The reward is the inverse of the orientation error between the object orientation and the goal orientation.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
        rot_eps: The threshold for the orientation error. Default is 1e-3.
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)

    return 1.0 / (dtheta + rot_eps)

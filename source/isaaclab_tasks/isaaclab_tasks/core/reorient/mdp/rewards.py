# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import ReorientCommand


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
    command_term: ReorientCommand = env.command_manager.get_term(command_name)

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # obtain the threshold for the orientation error
    threshold = command_term.cfg.orientation_success_threshold
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w.torch, goal_quat_w)

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
    command_term: ReorientCommand = env.command_manager.get_term(command_name)

    # obtain the goal position
    goal_pos_e = command_term.command[:, 0:3]
    # obtain the object position in the environment frame
    object_pos_e = asset.data.root_pos_w.torch - env.scene.env_origins

    return torch.linalg.norm(goal_pos_e - object_pos_e, ord=2, dim=-1)


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
    command_term: ReorientCommand = env.command_manager.get_term(command_name)

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w.torch, goal_quat_w)

    return 1.0 / (dtheta + rot_eps)


@torch.jit.script
def evaluate_reorient_success(
    object_quat: torch.Tensor, target_quat: torch.Tensor, success_tolerance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate reorientation success while exposing its physical error.

    This is the single per-step success evaluation: callers reuse the returned
    flags and errors for the reward, the episode bookkeeping, and the
    episode-minimum error tracking instead of recomputing the quaternion math.

    Args:
        object_quat: Object ``(x, y, z, w)`` orientations.
        target_quat: Target ``(x, y, z, w)`` orientations.
        success_tolerance: Maximum successful orientation error [rad].

    Returns:
        Per-environment success flags and orientation errors [rad].
    """
    orientation_error = math_utils.quat_error_magnitude(object_quat, target_quat)
    return orientation_error <= success_tolerance, orientation_error


@torch.jit.script
def reorient_reward(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    object_pos: torch.Tensor,
    target_pos: torch.Tensor,
    goal_reached: torch.Tensor,
    rotation_distance: torch.Tensor,
    actions: torch.Tensor,
    distance_scale: float,
    rotation_scale: float,
    rotation_epsilon: float,
    action_penalty_scale: float,
    success_bonus: float,
    fall_distance: float,
    fall_penalty: float,
    averaging_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Direct reorientation reward and success state transition.

    The success evaluation is not recomputed here: callers pass the flags and
    orientation errors from :func:`evaluate_reorient_success`, computed once
    per step.

    Args:
        reset_buf: Current episode-reset flags.
        reset_goal_buf: Current goal-reset flags.
        successes: Goals reached in each episode.
        consecutive_successes: Moving-average success count.
        object_pos: Object positions in the environment frame [m].
        target_pos: Goal positions in the environment frame [m].
        goal_reached: Per-environment success flags for this step.
        rotation_distance: Per-environment orientation errors [rad].
        actions: Normalized joint actions.
        distance_scale: Position-distance reward scale [1/m].
        rotation_scale: Orientation reward scale [rad].
        rotation_epsilon: Orientation reward regularizer [rad].
        action_penalty_scale: Squared-action reward scale.
        success_bonus: Reward added when a goal is reached.
        fall_distance: Object-to-goal termination distance [m].
        fall_penalty: Reward added when the object is out of reach.
        averaging_factor: Consecutive-success moving-average factor.

    Returns:
        Reward, goal-reset flags, episode success counts, and moving-average
        consecutive successes.
    """
    goal_distance = torch.linalg.norm(object_pos - target_pos, ord=2, dim=-1)
    goal_resets = reset_goal_buf | goal_reached
    successes = successes + goal_resets
    fell = goal_distance >= fall_distance
    reward = (
        goal_distance * distance_scale
        + rotation_scale / (rotation_distance + rotation_epsilon)
        + actions.square().sum(dim=-1) * action_penalty_scale
        + goal_resets.to(goal_distance.dtype) * success_bonus
        + fell.to(goal_distance.dtype) * fall_penalty
    )
    resets = reset_buf | fell
    num_resets = resets.sum()
    finished_successes = (successes * resets).sum()
    mean_successes = finished_successes / num_resets.clamp_min(1)
    consecutive_successes = torch.where(
        num_resets > 0,
        averaging_factor * mean_successes + (1.0 - averaging_factor) * consecutive_successes,
        consecutive_successes,
    )
    return reward, goal_resets, successes, consecutive_successes

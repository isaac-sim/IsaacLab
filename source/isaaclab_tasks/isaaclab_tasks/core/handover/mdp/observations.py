# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the manager-based handover task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

# Handover reuses the reorientation fingertip observation terms verbatim.
from isaaclab_tasks.core.reorient.mdp.observations import (  # noqa: F401
    fingertip_pos,
    fingertip_quat,
    fingertip_vel,
)

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def hand_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Return one hand's Direct-compatible raw action across resets.

    Args:
        env: Environment containing the action term and episode-length buffer.
        action_name: Action term whose raw action is observed.

    Returns:
        Current raw actions, retaining pre-reset actions while episode length is zero.
    """
    raw_action = env.action_manager.get_term(action_name).raw_actions
    reset_actions = getattr(env, "_handover_reset_actions", None)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if reset_actions is None or action_name not in reset_actions or episode_length_buf is None:
        return raw_action
    return torch.where((episode_length_buf == 0).unsqueeze(-1), reset_actions[action_name], raw_action)


def object_goal(
    env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg, vel_obs_scale: float
) -> torch.Tensor:
    """Return the 24-dimensional object and handover-goal observation block.

    Position components use [m], linear velocities [m/s], angular velocities
    [rad/s], and quaternion components are unitless. The angular-velocity
    scale arrives as the ``vel_obs_scale`` term param, wired at declaration.

    Args:
        env: Environment containing the object and goal command.
        command_name: Goal command term name.
        object_cfg: Object scene entity.
        vel_obs_scale: Angular-velocity observation scale.

    Returns:
        Object pose, spatial velocity, goal pose, and quaternion error, shape ``(num_envs, 24)``.
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    object_pos_e = object_asset.data.root_pos_w.torch - env.scene.env_origins
    object_quat = object_asset.data.root_quat_w.torch
    quat_error = math_utils.quat_mul(object_quat, math_utils.quat_conjugate(command_term.quat_command_w))
    return torch.cat(
        (
            object_pos_e,
            object_quat,
            object_asset.data.root_lin_vel_w.torch,
            vel_obs_scale * object_asset.data.root_ang_vel_w.torch,
            command_term.pos_command_e,
            command_term.quat_command_w,
            quat_error,
        ),
        dim=-1,
    )

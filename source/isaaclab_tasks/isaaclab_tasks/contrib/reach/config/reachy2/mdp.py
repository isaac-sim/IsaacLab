# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP terms for the Reachy 2 reach tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.envs.mdp import UniformPoseCommand


def reset_joint_targets_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the joint position targets to the default joint positions.

    Joints without an action term otherwise track a zero target, which drags the idle arm,
    neck, and grippers away from their configured pose. Action terms overwrite the targets
    of the joints they command on every step.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset. ``None`` resets all environments.
        asset_cfg: The articulation to update.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos.torch
    if env_ids is not None:
        default_joint_pos = default_joint_pos[env_ids]
    asset.actuators.target_command.set_position_index(value=default_joint_pos.clone(), env_ids=env_ids)


def pose_commands_success(env: ManagerBasedRLEnv, command_names: list[str]) -> torch.Tensor:
    """Terminate environments where every listed pose command satisfies its success thresholds.

    Args:
        env: The environment instance.
        command_names: Names of the uniform pose command terms that must all succeed.

    Returns:
        A boolean tensor of shape ``(num_envs,)`` that is true where all commands succeed.
    """
    success = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for command_name in command_names:
        command: UniformPoseCommand = env.command_manager.get_term(command_name)
        success &= command.compute_success()
    return success

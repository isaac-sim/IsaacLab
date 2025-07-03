# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def weighted_generated_commands(
    env: ManagerBasedRLEnv, command_name: str, weights: dict[str, float] = None
) -> torch.Tensor:
    """Generate weighted components of a command.

    This function retrieves commands and applies weights to different command components.
    For example, in velocity commands, you might want to weight lin_vel_x differently
    from ang_vel_z.

    Prerequisites:
        The command term must have a 'command_names' property that lists all valid command
        components. If this property is missing, a RuntimeError will be raised.

    Args:
        env: Environment instance.
        command_name: Name of the command generator.
        weights: Dictionary mapping command components to their weights.
                For example: {"lin_vel_x": 0.5, "lin_vel_y": 0.8, "ang_vel_z": 1.2}
                Components not specified will use a default weight of 1.0.
                All specified components must exist in the command term's command_names.

    Returns:
        Weighted command tensor.

    Raises:
        RuntimeError: If the command term does not have the required 'cfg.command_names' property.
        ValueError: If any weight key is not found in the command term's command_names.

    Example:
        >>> weights = {"lin_vel_x": 0.5, "height": 2.0}
        >>> weighted_commands = weighted_generated_commands(env, "base_velocity", weights)
    """
    # Get the command generator
    command_term = env.command_manager.get_term(command_name)

    # Check if command_names exists
    if not hasattr(command_term.cfg, "command_names"):
        raise RuntimeError(
            f"Command term '{command_name}' does not have 'command_names' property. "
            "This property is required for weighted command generation."
        )

    # Get the raw commands
    commands = env.command_manager.get_command(command_name)

    # If no weights specified, return raw commands
    if weights is None:
        return commands

    # Verify all weight keys exist in command_names
    if invalid_keys := set(weights.keys()) - set(command_term.cfg.command_names):
        raise ValueError(
            f"Invalid weight keys found: {invalid_keys}. Valid command components are: {command_term.cfg.command_names}"
        )

    # Create weight tensor matching command shape, initialized to 1.0
    weight_tensor = torch.ones_like(commands)

    # Apply weights for each command component
    idx = 0
    for component_name in command_term.cfg.command_names:
        if component_name in weights:
            weight_tensor[:, idx] = weights[component_name]
            idx += 1

    # Apply weights to commands
    weighted_commands = commands * weight_tensor

    return weighted_commands


def upper_body_last_action(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Extract the last action of the upper body."""
    asset = env.scene[asset_cfg.name]
    joint_pos_target = asset.data.joint_pos_target
    upper_body_joint_indices = asset.actuators["arms"]._joint_indices

    # Get upper body joint positions for all environments
    upper_body_joint_pos_target = joint_pos_target[:, upper_body_joint_indices]

    return upper_body_joint_pos_target

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foot- and head-specific reference tracking without contact labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_feet_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, body_names: list[str], std: float
) -> torch.Tensor:
    """Average per-foot XYZ tracking rewards after horizontal position and yaw alignment.

    Each selected foot contributes ``exp(-squared_position_error / std**2)``
    independently, without dilution by other tracked bodies. The reference
    retains its world height; ``std`` is measured in meters.
    """
    command = env.command_manager.get_term(command_name)
    body_indices = [command.cfg.body_names.index(name) for name in body_names]
    error = (
        (command.body_pos_relative_w[:, body_indices] - command.robot_body_pos_w[:, body_indices]).square().sum(dim=-1)
    )
    return torch.exp(-error / std**2).mean(dim=-1)


def motion_head_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, body_name: str, std: float
) -> torch.Tensor:
    """Reward head XYZ tracking independently of other bodies, with ``std`` in meters.

    The reference uses the same horizontal position and yaw alignment as the
    whole-body reward, retaining its height and tilt.
    """
    command = env.command_manager.get_term(command_name)
    body_index = command.cfg.body_names.index(body_name)
    error = (command.body_pos_relative_w[:, body_index] - command.robot_body_pos_w[:, body_index]).square().sum(dim=-1)
    return torch.exp(-error / std**2)

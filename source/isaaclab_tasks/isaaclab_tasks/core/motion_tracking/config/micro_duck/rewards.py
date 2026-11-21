# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference position tracking for selected bodies without contact labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, body_names: list[str], std: float
) -> torch.Tensor:
    """Average per-body XYZ tracking rewards after horizontal position and yaw alignment.

    Each selected body contributes ``exp(-squared_position_error / std**2)``
    independently, without dilution by other tracked bodies. The reference
    retains its world height; ``std`` is measured in meters.
    """
    command = env.command_manager.get_term(command_name)
    body_indices = [command.cfg.body_names.index(name) for name in body_names]
    error = (
        (command.body_pos_relative_w[:, body_indices] - command.robot_body_pos_w[:, body_indices]).square().sum(dim=-1)
    )
    return torch.exp(-error / std**2).mean(dim=-1)

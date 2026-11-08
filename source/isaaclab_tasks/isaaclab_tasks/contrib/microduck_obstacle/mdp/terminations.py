# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck obstacle-navigation terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obstacle_collision(
    env: ManagerBasedRLEnv, sensor_name: str = "obstacle_contacts", threshold: float = 0.1
) -> torch.Tensor:
    """Detect obstacle-only contact in physics-step history; floor contacts are excluded."""
    history = env.scene[sensor_name].data.force_matrix_w_history
    if history is None:
        raise ValueError("Obstacle collision detection requires a filtered contact sensor with force history.")
    return (history.torch.norm(dim=-1) > threshold).flatten(start_dim=1).any(dim=-1)


def field_boundary(env: ManagerBasedRLEnv, extent: float = 3.0) -> torch.Tensor:
    """Truncate episodes outside the finite obstacle field, without calling it a collision."""
    relative = env.scene["robot"].data.root_link_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return relative.norm(dim=-1) > extent

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the drone navigation environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_obstacle_curriculum_state(
    env: ManagerBasedRLEnv,
    min_difficulty: int = 2,
    max_difficulty: int = 10,
) -> dict:
    """Get or lazily initialize the obstacle curriculum state dictionary.

    This helper function manages the obstacle curriculum state by initializing it
    on first call and returning the existing state on subsequent calls. The state
    is stored as a private attribute on the environment instance.

    The curriculum state tracks per-environment difficulty levels used to control
    the number of obstacles spawned in each environment. Difficulty progresses
    based on agent performance (successful goal reaching vs. collisions).

    Args:
        env: The manager-based RL environment instance.
        min_difficulty: Minimum difficulty level for obstacle density. Lower values
            mean fewer obstacles. Defaults to 2.
        max_difficulty: Maximum difficulty level for obstacle density. Higher values
            mean more obstacles. Defaults to 10.

    Returns:
        Dictionary containing:
            - "difficulty_levels": torch.Tensor of shape (num_envs,) with per-environment
              difficulty levels, initialized to min_difficulty
            - "min_difficulty": int, the minimum difficulty value
            - "max_difficulty": int, the maximum difficulty value

    Note:
        This function stores state directly on the environment as `_obstacle_curriculum_state`.
        It's designed to be called from both curriculum terms and reward functions to ensure
        consistent access to the difficulty state.
    """

    if not hasattr(env, "_obstacle_curriculum_state"):
        env._obstacle_curriculum_state = {
            "difficulty_levels": torch.ones(env.num_envs, device=env.device) * min_difficulty,
            "min_difficulty": min_difficulty,
            "max_difficulty": max_difficulty,
        }
    return env._obstacle_curriculum_state


def obstacle_density_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
    max_difficulty: int = 10,
    min_difficulty: int = 2,
) -> float:
    """Curriculum that adjusts obstacle density based on performance.

    The difficulty state is managed centrally via get_obstacle_curriculum_state().
    This ensures consistent access across curriculum, reward, and event terms.
    """
    # Get or initialize curriculum state
    curriculum_state = get_obstacle_curriculum_state(env, min_difficulty, max_difficulty)

    # Extract robot and command
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins
    position_error = torch.norm(target_position_w[env_ids] - current_position[env_ids], dim=1)

    # Decide difficulty changes
    crashed = env.termination_manager.terminated[env_ids]
    move_up = position_error < 1.5  # Success
    move_down = crashed & ~move_up

    # Update difficulty levels
    difficulty_levels = curriculum_state["difficulty_levels"]
    difficulty_levels[env_ids] += move_up.long() - move_down.long()
    difficulty_levels[env_ids] = torch.clamp(
        difficulty_levels[env_ids], min=curriculum_state["min_difficulty"], max=curriculum_state["max_difficulty"] - 1
    )

    return difficulty_levels.float().mean().item()

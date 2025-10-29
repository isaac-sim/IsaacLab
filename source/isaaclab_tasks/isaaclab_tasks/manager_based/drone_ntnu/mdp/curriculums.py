# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the drone navigation environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "target_pose"
) -> torch.Tensor:
    """Curriculum based on the distance the drone is from the target position at the end of the episode.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()

    current_position = asset.data.root_pos_w - env.scene.env_origins
    position_error = torch.norm(target_position_w[env_ids] - current_position[env_ids], dim=1)

    # robots that are within 1m range should progress to harder terrains
    move_up = position_error < 1.5
    move_down = env.termination_manager.terminated[env_ids]
    # update terrain levels
    terrain.update_drone_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def get_obstacle_curriculum_state(
    env: ManagerBasedRLEnv,
    min_difficulty: int = 2,
    max_difficulty: int = 10,
) -> dict:
    """Get or initialize the obstacle curriculum state.
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
        difficulty_levels[env_ids],
        min=curriculum_state["min_difficulty"],
        max=curriculum_state["max_difficulty"] - 1
    )
    
    return difficulty_levels.float().mean().item()
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

def obstacle_density_curriculum(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
    max_difficulty: int = 10,
    min_difficulty: int = 2,
) -> float:
    """Curriculum that adjusts obstacle density based on performance."""
    # Initialize
    if not hasattr(env, "_obstacle_difficulty_levels"):
        env._obstacle_difficulty_levels = torch.ones(env.num_envs, device=env.device) * min_difficulty
        env._min_obstacle_difficulty = min_difficulty 
        env._max_obstacle_difficulty = max_difficulty
        env._obstacle_difficulty = float(min_difficulty)
    
    if len(env_ids) == 0:
        return env._obstacle_difficulty
    
    # Extract robot
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Performance metric
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins
    position_error = torch.norm(target_position_w[env_ids] - current_position[env_ids], dim=1)
    
    # Decide difficulty changes
    crashed = env.termination_manager.terminated[env_ids]
    move_up = position_error < 1.5  # Success
    move_down = crashed
    move_down *= ~move_up
    
    if env_ids[0] == 0 and (env.episode_length_buf[0] % 100 == 0 or move_up.any() or move_down.any()):
        print(f"\n=== Curriculum Update Debug ===")
        print(f"Step: {env.episode_length_buf[0].item()}")
        print(f"Current difficulty: {env._obstacle_difficulty_levels[env_ids[0]].item():.0f} / {max_difficulty}")
        print(f"Position error: {position_error[0].item():.2f}m (threshold: 1.5m)")
        print(f"Crashed: {crashed[0].item()}")
        print(f"Move up: {move_up.sum().item()} envs, Move down: {move_down.sum().item()} envs")
        if move_up[0]:
            print(f"  → Env 0 SUCCESS! Increasing difficulty")
        if move_down[0]:
            print(f"  → Env 0 FAILED! Decreasing difficulty")
        print("===============================\n")
        
    env._obstacle_difficulty_levels[env_ids] += 1 * move_up - 1 * move_down
    env._obstacle_difficulty_levels[env_ids] = torch.clip(env._obstacle_difficulty_levels[env_ids], 
                                                          min=env._min_obstacle_difficulty, 
                                                          max=env._max_obstacle_difficulty - 1)
    
    return env._obstacle_difficulty_levels.float().mean()
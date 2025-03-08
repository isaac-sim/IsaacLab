# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

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
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def command_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str, max_curriculum: float = 1.0
) -> None:
    """Curriculum based on the tracking reward of the robot when commanded to move at a desired velocity.

    This term is used to increase the range of commands when the robot's tracking reward is above 80% of the
    maximum.

    Returns:
        The cumulative increase in velocity command range.
    """
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    delta_range = torch.tensor([-0.1, 0.1], device=env.device)
    if not hasattr(env, "delta_lin_vel"):
        env.delta_lin_vel = torch.tensor(0.0, device=env.device)
    # If the tracking reward is above 80% of the maximum, increase the range of commands
    if torch.mean(episode_sums[env_ids]) / env.max_episode_length > 0.8 * reward_term_cfg.weight:
        lin_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        lin_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        base_velocity_ranges.lin_vel_x = torch.clamp(lin_vel_x + delta_range, -max_curriculum, max_curriculum).tolist()
        base_velocity_ranges.lin_vel_y = torch.clamp(lin_vel_y + delta_range, -max_curriculum, max_curriculum).tolist()
        env.delta_lin_vel = torch.clamp(env.delta_lin_vel + delta_range[1], 0.0, max_curriculum)
    return env.delta_lin_vel

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.orbit.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.orbit.envs.rl_env import RLEnv


def terrain_levels_vel(env: RLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.orbit.terrains.TerrainImporter` class.

    Returns:
        torch.Tensor: The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(env.command_manager.command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

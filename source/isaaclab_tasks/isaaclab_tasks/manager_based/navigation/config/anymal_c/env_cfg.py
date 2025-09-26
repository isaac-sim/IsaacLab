# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.navigation.navigation_env_cfg import RayCasterNavEnvCfg, TiledNavEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


def _play_init(self):
    # make a smaller scene for play
    self.scene.num_envs = 50
    self.scene.env_spacing = 2.5
    # spawn the robot randomly in the grid (instead of their terrain levels)
    self.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    self.observations.policy.enable_corruption = False


@configclass
class AnymalCRayCasterNavEnvCfg(RayCasterNavEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-c
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # turn off the self-collisions
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False


@configclass
class AnymalCTiledNavEnvCfg(TiledNavEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-c
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # turn off the self-collisions
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False


@configclass
class AnymalCRayCasterNavEnvCfg_PLAY(AnymalCRayCasterNavEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        _play_init(self)


@configclass
class AnymalCTiledNavEnvCfg_PLAY(AnymalCTiledNavEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        _play_init(self)

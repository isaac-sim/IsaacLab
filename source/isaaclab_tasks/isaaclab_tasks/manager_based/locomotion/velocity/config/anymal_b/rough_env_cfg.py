# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventsCfg,
    LocomotionVelocityRoughEnvCfg,
    StartupEventsCfg,
)
from isaaclab_tasks.utils import PresetCfg

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_B_CFG  # isort: skip


@configclass
class AnymalBPhysxEventsCfg(EventsCfg, StartupEventsCfg):
    pass


@configclass
class AnymalBEventsCfg(PresetCfg):
    default = AnymalBPhysxEventsCfg()
    newton = EventsCfg()
    physx = default


@configclass
class AnymalBRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    events: AnymalBEventsCfg = AnymalBEventsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-b
        self.scene.robot = ANYMAL_B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AnymalBRoughEnvCfg_PLAY(AnymalBRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

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
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

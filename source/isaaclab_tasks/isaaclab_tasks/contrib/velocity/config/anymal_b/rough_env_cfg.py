# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RoughPhysicsCfg,
    make_feather_pgs_physics_cfg,
)
from isaaclab_tasks.utils import preset

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_B_CFG  # isort: skip


@configclass
class PhysicsCfg(RoughPhysicsCfg):
    feather_pgs = make_feather_pgs_physics_cfg(
        pgs_iterations=16, pgs_beta=0.02, pgs_cfm=1.0e-5, pgs_omega=0.8, angular_damping=0.5
    )


@configclass
class AnymalBRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-b
        self.scene.robot = ANYMAL_B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["legs"].armature = preset(default=0.0, feather_pgs=0.05)


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

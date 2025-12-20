# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

from .env_cfg import (
    G1ActionsCfg, 
    G1ObservationsCfg, 
    G1RewardsCfg, 
    G1SceneCfg,
    G1TerminationsCfg,
    G1CurriculumCfg, 
    G1EventCfg,
    G1CommandsCfg, 
)


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1RewardsCfg = G1RewardsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    observations: G1ObservationsCfg = G1ObservationsCfg()
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    events: G1EventCfg = G1EventCfg()
    commands: G1CommandsCfg = G1CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # gait duration in sec
        self.phase_dt = 0.3 * 2

        # # physics dt
        # self.sim.dt = 0.002 # 500 Hz
        # self.decimation = 10 # 50 Hz
        # self.sim.render_interval = self.decimation

        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),    
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.push_robot = None

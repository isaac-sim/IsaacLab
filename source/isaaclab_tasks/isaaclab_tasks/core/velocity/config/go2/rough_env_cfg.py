# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.utils import preset

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["base_legs"].armature = preset(default=0.0, newton_mjwarp=0.02)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["boxes"].grid_height_range = (0.025, 0.1)
        terrains["random_rough"].noise_range = (0.01, 0.06)
        terrains["random_rough"].noise_step = 0.01
        terrains["pyramid_stairs"].step_height_range = (0.025, 0.12)
        terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.12)
        # actions
        self.actions.joint_pos.scale = 0.25
        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

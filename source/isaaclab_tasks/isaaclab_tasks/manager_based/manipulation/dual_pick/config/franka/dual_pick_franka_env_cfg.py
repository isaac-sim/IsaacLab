# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.dual_pick.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.dual_pick.dual_pick_env_cfg import (
    DualPickEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaDualPickEnvCfg(DualPickEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka robots
        self.scene.robot_left = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotLeft",
            init_state=FRANKA_PANDA_CFG.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.037,
                    "panda_joint7": 0.741,
                },
                joint_vel={".*": 0.0},
                pos=[0.0, 0.3, 0.0],
            ),
        )

        self.scene.robot_right = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotRight",
            init_state=FRANKA_PANDA_CFG.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.037,
                    "panda_joint7": 0.741,
                },
                joint_vel={".*": 0.0},
                pos=[0.0, -0.3, 0.0],
            ),
        )

        # Set actions for both arms
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot_left",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot_right",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )


@configclass
class FrankaDualPickEnvCfg_PLAY(FrankaDualPickEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

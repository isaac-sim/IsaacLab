# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.unimanual.reach_openarm_uni_env_cfg import (
    ReachEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

##
# Environment configuration
##


@configclass
class OpenArmReachEnvCfg(ReachEnvCfg):
    """Configuration for the single-arm OpenArm Reach Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to OpenArm
        self.scene.robot = OPENARM_UNI_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_joint1": 1.57,
                    "openarm_joint2": 0.0,
                    "openarm_joint3": -1.57,
                    "openarm_joint4": 1.57,
                    "openarm_joint5": 0.0,
                    "openarm_joint6": 0.0,
                    "openarm_joint7": 0.0,
                    "openarm_finger_joint.*": 0.0,
                },  # Close the gripper
            ),
        )

        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["openarm_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["openarm_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["openarm_hand"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_joint.*",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "openarm_hand"


@configclass
class OpenArmReachEnvCfg_PLAY(OpenArmReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

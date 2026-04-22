# Copyright (c) 2026-2027, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.deploy.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import FLEXIV_RIZON4S_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class Rizon4sReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events.robot_joint_stiffness_and_damping.params["asset_cfg"].joint_names = [
            "joint[1-2]",
            "joint[3-4]",
            "joint[5-7]",
        ]
        self.events.joint_friction.params["asset_cfg"].joint_names = ["joint[1-2]", "joint[3-4]", "joint[5-7]"]

        # switch robot to Flexiv Rizon 4s
        self.scene.robot = FLEXIV_RIZON4S_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Configure the end-effector frame relative to base frame for Rizon 4s
        self.rewards.end_effector_keypoint_tracking.params["asset_cfg"] = SceneEntityCfg("ee_frame_wrt_base_frame")
        self.rewards.end_effector_keypoint_tracking_exp.params["asset_cfg"] = SceneEntityCfg("ee_frame_wrt_base_frame")
        self.scene.ee_frame_wrt_base_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
            source_frame_offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/flange",
                    name="end_effector",
                ),
            ],
        )

        self.commands.ee_pose.debug_vis = True

        # Incremental joint position action configuration
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["joint[1-7]"], scale=0.0625, use_zero_offset=True
        )
        # override command generator body
        # end-effector is along z-direction for Rizon 4s
        self.target_pos_centre = (0.4, 0.0, 0.4)
        self.target_pos_range = (0.4, 0.4, 0.35)
        self.commands.ee_pose.body_name = "flange"
        self.commands.ee_pose.ranges.pos_x = (
            self.target_pos_centre[0] - self.target_pos_range[0],
            self.target_pos_centre[0] + self.target_pos_range[0],
        )
        self.commands.ee_pose.ranges.pos_y = (
            self.target_pos_centre[1] - self.target_pos_range[1],
            self.target_pos_centre[1] + self.target_pos_range[1],
        )
        self.commands.ee_pose.ranges.pos_z = (
            self.target_pos_centre[2] - self.target_pos_range[2],
            self.target_pos_centre[2] + self.target_pos_range[2],
        )

        self.target_rot_centre = (math.pi, 0.0, 0.0)  # end-effector facing down
        self.target_rot_range = (math.pi / 2, math.pi / 2, math.pi)
        self.commands.ee_pose.ranges.roll = (
            self.target_rot_centre[0] - self.target_rot_range[0],
            self.target_rot_centre[0] + self.target_rot_range[0],
        )
        self.commands.ee_pose.ranges.pitch = (
            self.target_rot_centre[1] - self.target_rot_range[1],
            self.target_rot_centre[1] + self.target_rot_range[1],
        )
        self.commands.ee_pose.ranges.yaw = (
            self.target_rot_centre[2] - self.target_rot_range[2],
            self.target_rot_centre[2] + self.target_rot_range[2],
        )


@configclass
class Rizon4sReachEnvCfg_PLAY(Rizon4sReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

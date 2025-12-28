# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

@configclass
class G1ActionsCfg:
    """
    Action specifications for the MDP.
    Joint order is aligned with hardware motor order.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            "left_hip_pitch_joint", 
            "left_hip_roll_joint", 
            "left_hip_yaw_joint", 
            "left_knee_joint", 
            "left_ankle_pitch_joint", 
            "left_ankle_roll_joint", 
            
            "right_hip_pitch_joint", 
            "right_hip_roll_joint", 
            "right_hip_yaw_joint", 
            "right_knee_joint", 
            "right_ankle_pitch_joint", 
            "right_ankle_roll_joint", 
            
            "waist_yaw_joint", 
            "waist_roll_joint", 
            "waist_pitch_joint", 
            
            "left_shoulder_pitch_joint", 
            "left_shoulder_roll_joint", 
            "left_shoulder_yaw_joint", 
            "left_elbow_joint", 
            "left_wrist_roll_joint", 
            "left_wrist_pitch_joint", 
            "left_wrist_yaw_joint", 

            "right_shoulder_pitch_joint", 
            "right_shoulder_roll_joint", 
            "right_shoulder_yaw_joint", 
            "right_elbow_joint", 
            "right_wrist_roll_joint", 
            "right_wrist_pitch_joint", 
            "right_wrist_yaw_joint", 
        ],
        scale=0.25,
        use_default_offset=True, 
        preserve_order=True,
        )
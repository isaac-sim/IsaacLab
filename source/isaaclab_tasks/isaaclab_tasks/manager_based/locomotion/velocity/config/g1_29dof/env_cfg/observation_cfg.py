# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_29dof.mdp as g1_mdp

"""
leggedlab
"""

@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel, 
        noise=Unoise(n_min=-0.2, n_max=0.2), 
        scale=0.25,
        ) 
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel, 
        noise=Unoise(n_min=-0.01, n_max=0.01), 
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
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
                    preserve_order=True,
                ), 
                },
        ) 
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel, 
        noise=Unoise(n_min=-1.5, n_max=1.5), 
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
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
                    preserve_order=True,
                ), 
                },
        scale=0.05,
        ) 
    actions = ObsTerm(func=mdp.last_action)
    height_scan = ObsTerm(
        func=mdp.height_scan, 
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        # self.history_length = 5 # unitree_rl_lab uses 5
        self.history_length = 10 # legged_lab uses 10

@configclass
class CriticCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel, 
        scale=0.25,
        ) 
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel, 
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
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
                    preserve_order=True,
                ), 
                },
        ) 
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
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
                    preserve_order=True,
                ), 
                },
        scale=0.05, 
        )
    actions = ObsTerm(func=mdp.last_action)
    height_scan = ObsTerm(
        func=mdp.height_scan, 
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    # privileged observations
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    feet_contact = ObsTerm(
        func=vel_mdp.foot_contact, 
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}, 
        )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        # self.history_length = 5 # unitree_rl_lab uses 5
        self.history_length = 10 # legged_lab uses 10

    
# @configclass
# class LoggingObsCfg(ObsGroup):
#     """Observations for logging purposes."""

#     base_lin_vel = ObsTerm(func=mdp.base_lin_vel) 
#     base_ang_vel = ObsTerm(func=mdp.base_ang_vel) 
#     velocity_commands = ObsTerm(
#         func=mdp.generated_commands,
#         params={"command_name": "base_velocity"},
#     )
    

#     def __post_init__(self):
#         self.enable_corruption = False
#         self.concatenate_terms = True

@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    # logging: LoggingObsCfg = LoggingObsCfg()
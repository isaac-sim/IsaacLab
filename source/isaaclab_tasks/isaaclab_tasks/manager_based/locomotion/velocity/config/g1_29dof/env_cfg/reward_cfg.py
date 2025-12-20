# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_29dof.mdp as g1_mdp


"""
Legacy Walking policy
"""
@configclass
class G1RewardsCfg:
    """Reward terms for the MDP."""

    """
    task rewards
    """
    track_lin_vel_xy = RewTerm(
        func=vel_mdp.track_lin_vel_xy_yaw_frame_exp, 
        weight=2.0, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)} 
    )
    track_ang_vel_z = RewTerm(
        func=vel_mdp.track_ang_vel_z_world_exp, 
        weight=2.0, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)} 
    )
    
    """
    style rewards
    """
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_rate_l2_lower_body = RewTerm(
        func=g1_mdp.action_rate_l2, 
        weight=-0.01, 
        params={"joint_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]} # mjc order
        )
    action_rate_l2_upper_body = RewTerm(
        func=g1_mdp.action_rate_l2, 
        weight=-0.05, 
        params={"joint_idx": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]} # mjc order
        )

    # -- base penalties
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.75})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0) 
    body_orientation_l2 = RewTerm(
        func=g1_mdp.body_orientation_l2, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, 
        weight=-2.0
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0) 
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # -- contact penalties
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )

    """
    joint regularization.
    """
    energy = RewTerm(func=g1_mdp.energy, weight=-1e-3)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4) # optional
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    
    # penalize joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    joint_deviation = RewTerm(
        func=g1_mdp.variable_posture, 
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_velocity",
            "weight_standing": {
                ".*": 0.2, 
            },  
            "weight_walking": {
                # leg
                ".*hip_pitch.*": 0.02, 
                ".*hip_roll.*": 0.15,
                ".*hip_yaw.*": 0.15,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                # waist
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 0.1,
                ".*waist_pitch.*": 0.1, 
                # arms
                ".*shoulder_pitch.*": 0.15,
                ".*elbow.*": 0.1,
                ".*shoulder_roll.*": 0.15,
                ".*shoulder_yaw.*": 0.15,
                ".*wrist.*": 0.15,
            }, 
            "weight_running": {
                # leg
                ".*hip_pitch.*": 0.02, 
                ".*hip_roll.*": 0.15,
                ".*hip_yaw.*": 0.15,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                # waist
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 0.1,
                ".*waist_pitch.*": 0.1, 
                # arms
                ".*shoulder_pitch.*": 0.15,
                ".*elbow.*": 0.1,
                ".*shoulder_roll.*": 0.15,
                ".*shoulder_yaw.*": 0.15,
                ".*wrist.*": 0.15,
            }, 
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
        }
    )
    
    """
    foot orientation
    """

    # -- foot orientation penalities
    feet_yaw_diff = RewTerm(
        func=g1_mdp.reward_feet_yaw_diff, 
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        }
    )

    feet_yaw_mean = RewTerm(
        func=g1_mdp.reward_feet_yaw_mean, 
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        }
    )

    feet_roll = RewTerm(
        func=g1_mdp.reward_feet_roll,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        },
    )

    feet_roll_diff = RewTerm(
        func=g1_mdp.reward_feet_roll_diff,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        },
    )

    feet_pitch = RewTerm(
        func=g1_mdp.reward_feet_pitch,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        },
    )

    feet_pitch_diff = RewTerm(
        func=g1_mdp.reward_feet_pitch_diff,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        },
    )

    """
    gait
    """

    # rewarded when agent walks with single stance gait. 
    # this is sparse reward as agent receives reward equivalent to swing time only during single stance mode.
    feet_air_time = RewTerm(
        func=vel_mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 0.5,
        },
    )

    # for faster gait learning
    feet_swing = RewTerm(
        func=g1_mdp.reward_feet_swing,
        weight=2.0,
        params={
            # swing period in phase (0.5 is maximum)
            "swing_period": 0.2, 
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_roll.*"
            ),
            "cmd_threshold": 0.05, # or 0.1
            "command_name": "base_velocity",
        },
    )

    fly = RewTerm(
        func=g1_mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )

    """
    stance foot
    """

    feet_slide = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # penalize lateral foot distance
    foot_distance = RewTerm(
        func = g1_mdp.reward_foot_distance,
        weight=-2.0,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*_ankle_roll_link",
                preserve_order=True,
            ),
        },
    )

    feet_force = RewTerm(
        func=g1_mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    
    """
    swing foot
    """
    
    # optional: encourage specific foot clearance value 
    foot_clearance = RewTerm(
        func=g1_mdp.foot_clearance_reward,
        weight=2.0,
        params={
            "target_height": 0.12,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "standing_position_foot_z": 0.03539,
        },
    )
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for the G1 29-DoF granular locomotion task.

Cloned from the soft-contact task. A granular bed exerts no force on a rigid contact sensor, so
the gait terms are rebuilt on the contact state that the coupler feeds back from the MPM entry to
the proxy feet. The contact impulse term penalizes touchdown force magnitude using this same
contact state. Terms requiring per-shape contact data, ``undesired_contacts`` and ``feet_slide``, are dropped.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .. import mdp as g1_mdp
from .. import mdp as mpm_mdp
from .scene_cfg import SAND_DEPTH


@configclass
class G1RewardsCfg:
    """Reward terms for the MDP."""

    # metric_sliderbar = RewTerm(
    #     func=mpm_mdp.metric_sliderbar,
    #     weight=1.0,  # Zero-valued diagnostic; a zero weight would skip evaluation.
    #     params={
    #         "obs_group_name": "privileged",
    #         "obs_term_names": [
    #             # "base_lin_vel",
    #             "foot_height",
    #             "foot_contact",
    #             "foot_contact_force",
    #             "foot_air_time",
    #             # "terrain_material_parameters",
    #         ],
    #     },
    # )

    """
    task rewards
    """
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_heading = RewTerm(
        func=g1_mdp.track_heading_world_exp,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )

    """
    style rewards
    """
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_rate_l2_lower_body = RewTerm(
        func=g1_mdp.action_rate_l2,
        weight=-0.01,
        params={"joint_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    )
    action_rate_l2_upper_body = RewTerm(
        func=g1_mdp.action_rate_l2,
        weight=-0.05,
        params={"joint_idx": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]},
    )

    # -- base penalties
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10,
        params={"target_height": 0.75, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    """
    joint regularization.
    """
    energy = RewTerm(func=g1_mdp.energy, weight=-1e-3)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # penalize joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_deviation = RewTerm(
        func=g1_mdp.variable_posture_l1,  # type: ignore
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_velocity",
            "weight_standing": {
                # Allow the legs to adapt to the terrain while retaining upper-body posture.
                ".*hip_pitch.*": 0.02,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.08,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                ".*(waist|shoulder|elbow|wrist).*": 0.5,
            },
            "weight_walking": {
                # leg
                ".*hip_pitch.*": 0.02,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.08,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                # waist
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 2.0,
                # arms
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.25,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_yaw.*": 0.35,
                ".*wrist.*": 0.5,
            },
            "weight_running": {
                # leg
                ".*hip_pitch.*": 0.005,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.08,
                ".*knee.*": 0.005,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                # waist
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 2.0,
                # arms
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.25,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_yaw.*": 0.35,
                ".*wrist.*": 0.5,
            },
            "walking_threshold": 0.05,
            "running_threshold": 2.0,
        },
    )

    # """
    # foot orientation
    # """

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

    # feet_pitch = RewTerm(
    #     func=g1_mdp.reward_feet_pitch,
    #     weight=-4.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=[".*ankle_roll.*"],
    #             preserve_order=True,
    #         ),
    #     },
    # )

    # feet_pitch_diff = RewTerm(
    #     func=g1_mdp.reward_feet_pitch_diff,
    #     weight=-4.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=[".*ankle_roll.*"],
    #             preserve_order=True,
    #         ),
    #     },
    # )
    # # avoid toe contact
    # feet_pitch_contact = RewTerm(
    #     func=mpm_mdp.feet_pitch_contact,
    #     weight=-4.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=[".*ankle_roll.*"],
    #             preserve_order=True,
    #         ),
    #     },
    # )

    stance_foot_angle_penalty = RewTerm(
        func=mpm_mdp.stance_foot_angle_penalty,  # type: ignore
        # weight=-4.0,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "angle_tolerance": math.radians(0.0),
        },
    )

    """
    gait
    """

    feet_air_time = RewTerm(
        func=mpm_mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "velocity_threshold": 0.05,
        },
    )

    break_contact = RewTerm(
        func=mpm_mdp.break_contact_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "recovery_linear_velocity": 0.2,
            "recovery_angular_velocity": 0.5,
            "recovery_tilt": 0.2,
        },
    )

    no_fly = RewTerm(
        func=mpm_mdp.no_fly,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "velocity_threshold": 1.0,
        },
    )

    """
    Stance foot
    """
    # penalize lateral foot distance
    foot_distance = RewTerm(
        func=g1_mdp.reward_foot_lateral_symmetry,
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

    contact_impulse = RewTerm(
        func=mpm_mdp.reward_soft_landing,
        weight=-5e-3,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
    )

    """
    Swing foot
    """

    foot_clearance = RewTerm(
        func=mpm_mdp.foot_clearance_reward,
        weight=5.0,
        params={
            "target_height": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "standing_position_foot_z": 0.03539,
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "ground_height_offset": SAND_DEPTH,
        },
    )

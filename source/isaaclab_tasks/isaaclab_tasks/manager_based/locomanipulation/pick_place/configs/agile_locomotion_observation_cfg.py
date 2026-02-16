# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp import observations as locomanip_obs


@configclass
class AgileTeacherPolicyObservationsCfg(ObsGroup):
    """Observation specifications for the Agile lower body policy.

    Note: This configuration defines only part of the observation input to the Agile lower body policy.
    The lower body command portion is appended to the observation tensor in the action term, as that
    is where the environment has access to those commands.
    """

    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        scale=1.0,
    )

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                # Agile policy was trained in Isaac Lab + PhysX which has different joint ordering than MuJoCo
                # this list does the re-mapping of joint orders
                joint_names=[
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                    "waist_yaw_joint",
                    "left_hip_roll_joint",
                    "right_hip_roll_joint",
                    "waist_roll_joint",
                    "left_hip_yaw_joint",
                    "right_hip_yaw_joint",
                    "waist_pitch_joint",
                    "left_knee_joint",
                    "right_knee_joint",
                    "left_shoulder_pitch_joint",
                    "right_shoulder_pitch_joint",
                    "left_ankle_pitch_joint",
                    "right_ankle_pitch_joint",
                    "left_shoulder_roll_joint",
                    "right_shoulder_roll_joint",
                    "left_ankle_roll_joint",
                    "right_ankle_roll_joint",
                    "left_shoulder_yaw_joint",
                    "right_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "right_elbow_joint",
                    "left_wrist_roll_joint",
                    "right_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "right_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_wrist_yaw_joint",
                ],
                preserve_order=True,
            ),
        },
    )

    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                # Agile policy was trained in Isaac Lab + PhysX which has different joint ordering than MuJoCo
                # this list does the re-mapping of joint orders
                joint_names=[
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                    "waist_yaw_joint",
                    "left_hip_roll_joint",
                    "right_hip_roll_joint",
                    "waist_roll_joint",
                    "left_hip_yaw_joint",
                    "right_hip_yaw_joint",
                    "waist_pitch_joint",
                    "left_knee_joint",
                    "right_knee_joint",
                    "left_shoulder_pitch_joint",
                    "right_shoulder_pitch_joint",
                    "left_ankle_pitch_joint",
                    "right_ankle_pitch_joint",
                    "left_shoulder_roll_joint",
                    "right_shoulder_roll_joint",
                    "left_ankle_roll_joint",
                    "right_ankle_roll_joint",
                    "left_shoulder_yaw_joint",
                    "right_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "right_elbow_joint",
                    "left_wrist_roll_joint",
                    "right_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "right_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_wrist_yaw_joint",
                ],
                preserve_order=True,
            ),
        },
    )

    actions = ObsTerm(
        func=locomanip_obs.last_action_with_remap,
        scale=1.0,
        params={
            "action_name": "lower_body_joint_pos",
            "asset_cfg": SceneEntityCfg(
                "robot",
                # Agile policy was trained in Isaac Lab + PhysX which has different joint ordering than MuJoCo
                # this list does the re-mapping of joint orders
                joint_names=[
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "right_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "right_hip_yaw_joint",
                    "left_knee_joint",
                    "right_knee_joint",
                    "left_ankle_pitch_joint",
                    "right_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_ankle_roll_joint",
                ],
                preserve_order=True,
            ),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True

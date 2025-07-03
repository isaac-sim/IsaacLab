# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.mdp.observations as obs_mdp


@configclass
class ObservationsCfg(ObsGroup):
    """Observation specifications for the MDP."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        scale=1.0,
    )

    joint_pos_upper = ObsTerm(
        func=mdp.joint_pos_rel,
        scale=0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                    ".*_hand_.*",
                ],
            ),
        },
    )

    joint_pos_lower = ObsTerm(
        func=mdp.joint_pos_rel,
        scale=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "waist_.*_joint",
                ],
            ),
        },
    )

    joint_vel_upper = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                    ".*_hand_.*",
                ],
            ),
        },
    )

    joint_vel_lower = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "waist_.*_joint",
                ],
            ),
        },
    )

    last_actions_upper = ObsTerm(
        func=obs_mdp.upper_body_last_action,
        scale=0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                    ".*_hand_.*",
                ],
            ),
        },
    )

    last_actions_lower = ObsTerm(
        func=mdp.last_action,
        scale=1.0,
        params={
            "action_name": "lower_body_joint_pos",
        },
    )

    base_commands = ObsTerm(
        func=obs_mdp.weighted_generated_commands,
        params={
            "command_name": "base_velocity",
            "weights": {
                "lin_vel_x": 1.0,
                "lin_vel_y": 1.0,
                "height": 1.0,
            },
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StandingObservationsCfg:
    """Observation specifications for the standing behavior MDP.

    The upper body action will be randomly sampled, no observation for it.

    """

    @configclass
    class LowerBodyPolicyCfg(ObservationsCfg):
        """Observations for lower bodypolicy group."""

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5

    # observation groups
    policy: LowerBodyPolicyCfg = LowerBodyPolicyCfg()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach environment configurations for Pollen Robotics Reachy 2.

Environments:
* :class:`Reachy2RightReachEnvCfg`: Right arm reach task.
* :class:`Reachy2LeftReachEnvCfg`: Left arm reach task.
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import REACHY2_CFG  # isort: skip

##
# Right arm reach environment
##


@configclass
class Reachy2RightReachEnvCfg(ReachEnvCfg):
    """Reach environment for Reachy 2 right arm."""

    def __post_init__(self):
        super().__post_init__()

        # Reachy 2 is a standing humanoid — no table needed
        self.scene.table = None
        # Ground flush with robot base (base_link spawns at z=0)
        self.scene.ground.init_state.pos = (0.0, 0.0, 0.0)

        # Switch robot to Reachy 2
        self.scene.robot = REACHY2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    # Neck neutral
                    "neck_.*": 0.0,
                    # Right arm — slight resting pose
                    "r_shoulder_pitch": 0.0,
                    "r_shoulder_roll": -0.2,
                    "r_elbow_yaw": 0.0,
                    "r_elbow_pitch": -0.5,
                    "r_wrist_roll": 0.0,
                    "r_wrist_pitch": 0.0,
                    "r_wrist_yaw": 0.0,
                    # Left arm — tucked away from right arm workspace
                    "l_shoulder_pitch": 0.0,
                    "l_shoulder_roll": 0.5,
                    "l_elbow_yaw": 0.0,
                    "l_elbow_pitch": -1.0,
                    "l_wrist_.*": 0.0,
                    # Grippers open
                    ".*hand_finger.*": 0.0,
                },
            ),
        )

        # Right arm joints for actions and observations
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "r_shoulder_pitch",
                "r_shoulder_roll",
                "r_elbow_yaw",
                "r_elbow_pitch",
                "r_wrist_roll",
                "r_wrist_pitch",
                "r_wrist_yaw",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # End-effector body — right hand palm
        _ee = "r_hand_palm_link"
        self.rewards.end_effector_position_tracking.params["asset_cfg"] = SceneEntityCfg("robot", body_names=[_ee])
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[_ee]
        )
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"] = SceneEntityCfg("robot", body_names=[_ee])
        self.commands.ee_pose.body_name = _ee

        # Command ranges suited for a humanoid arm at ~0.6 m reach
        self.commands.ee_pose.ranges.pos_x = (0.3, 0.55)
        self.commands.ee_pose.ranges.pos_y = (-0.4, 0.0)
        self.commands.ee_pose.ranges.pos_z = (0.8, 1.2)
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class Reachy2RightReachEnvCfg_PLAY(Reachy2RightReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


##
# Left arm reach environment
##


@configclass
class Reachy2LeftReachEnvCfg(ReachEnvCfg):
    """Reach environment for Reachy 2 left arm."""

    def __post_init__(self):
        super().__post_init__()

        # Reachy 2 is a standing humanoid — no table needed
        self.scene.table = None
        # Ground flush with robot base
        self.scene.ground.init_state.pos = (0.0, 0.0, 0.0)

        self.scene.robot = REACHY2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "neck_.*": 0.0,
                    # Left arm — slight resting pose
                    "l_shoulder_pitch": 0.0,
                    "l_shoulder_roll": 0.2,
                    "l_elbow_yaw": 0.0,
                    "l_elbow_pitch": -0.5,
                    "l_wrist_roll": 0.0,
                    "l_wrist_pitch": 0.0,
                    "l_wrist_yaw": 0.0,
                    # Right arm — tucked away
                    "r_shoulder_pitch": 0.0,
                    "r_shoulder_roll": -0.5,
                    "r_elbow_yaw": 0.0,
                    "r_elbow_pitch": -1.0,
                    "r_wrist_.*": 0.0,
                    ".*hand_finger.*": 0.0,
                },
            ),
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "l_shoulder_pitch",
                "l_shoulder_roll",
                "l_elbow_yaw",
                "l_elbow_pitch",
                "l_wrist_roll",
                "l_wrist_pitch",
                "l_wrist_yaw",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        _ee = "l_hand_palm_link"
        self.rewards.end_effector_position_tracking.params["asset_cfg"] = SceneEntityCfg("robot", body_names=[_ee])
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[_ee]
        )
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"] = SceneEntityCfg("robot", body_names=[_ee])
        self.commands.ee_pose.body_name = _ee

        # Mirror of right arm ranges
        self.commands.ee_pose.ranges.pos_x = (0.3, 0.55)
        self.commands.ee_pose.ranges.pos_y = (0.0, 0.4)
        self.commands.ee_pose.ranges.pos_z = (0.8, 1.2)
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class Reachy2LeftReachEnvCfg_PLAY(Reachy2LeftReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab_assets.robots.agility import ARM_JOINT_NAMES, LEG_JOINT_NAMES

from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.digit.rough_env_cfg import DigitRewards, DigitRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg


@configclass
class DigitLocoManipRewards(DigitRewards):
    joint_deviation_arms = None

    joint_vel_hip_yaw = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_hip_yaw"])},
    )

    left_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "command_name": "left_ee_pose",
        },
    )

    left_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )

    left_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "command_name": "left_ee_pose",
        },
    )

    right_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "command_name": "right_ee_pose",
        },
    )

    right_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "command_name": "right_ee_pose",
        },
    )


@configclass
class DigitLocoManipObservations:

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class DigitLocoManipCommands:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.25,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_arm_wrist_yaw",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.50),
            pos_y=(0.05, 0.50),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(math.pi / 2.0 - 0.1, math.pi / 2.0 + 0.1),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="right_arm_wrist_yaw",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.50),
            pos_y=(-0.50, -0.05),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(-math.pi / 2.0 - 0.1, -math.pi / 2.0 + 0.1),
        ),
    )


@configclass
class DigitEvents(EventCfg):
    # Add an external force to simulate a payload being carried.
    left_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    right_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )


@configclass
class DigitLocoManipEnvCfg(DigitRoughEnvCfg):
    rewards: DigitLocoManipRewards = DigitLocoManipRewards()
    observations: DigitLocoManipObservations = DigitLocoManipObservations()
    commands: DigitLocoManipCommands = DigitLocoManipCommands()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 14.0

        # Rewards:
        self.rewards.flat_orientation_l2.weight = -10.5
        self.rewards.termination_penalty.weight = -100.0

        # Change terrain to flat.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove height scanner.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # Remove terrain curriculum.
        self.curriculum.terrain_levels = None


class DigitLocoManipEnvCfg_PLAY(DigitLocoManipEnvCfg):

    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play.
        self.observations.policy.enable_corruption = False
        # Remove random pushing.
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None

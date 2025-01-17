# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.footstep.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.footstep.footstep_env_cfg import (
    LocomotionFootstepEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets import SIGMABAN_CFG  # isort: skip


@configclass
class SigmabanRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=6.0, # 4.0
        params={"command_name": "base_velocity", "std": 0.3}, # 0.5
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.0, #1.0
        params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0, #0.25
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    # feet_contact_penalty = RewTerm(
    #     func=mdp.feet_contact_penalty,
    #     weight=-0.25,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")
    #     },
    # )

    legs_contact_penalty = RewTerm(
        func=mdp.feet_contact_penalty,
        weight=-0.5, #-0.5
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot", ".*_tibia", ".*_knee", ".*_mx106_block_hip"])
        },
    )
    
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-1.0, #-1.0
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_.*")}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2, #-0.3
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2, #-0.2
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    # feet_normal_ground = RewTerm(
    #     func=mdp.feet_normal_ground,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    # )
    right_foot_normal_ground = RewTerm(
        func=mdp.feet_normal_ground,
        weight=-0.0, #-5.0
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_foot")},
    )
    left_foot_normal_ground = RewTerm(
        func=mdp.feet_normal_ground,
        weight=-0.0, #-5.0
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_foot")},
    )
    torso_height = RewTerm(
        func=mdp.torso_height,
          weight=0.0, #-10.0
          params={"target_height": 0.30, "sensor_cfg": SceneEntityCfg("robot")}
    )
    # joint_deviation_torso = RewTerm(
    #     func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    # )


@configclass
class SigmabanFootstepEnvCfg(LocomotionFootstepEnvCfg):
    rewards: SigmabanRewards = SigmabanRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = SIGMABAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso"]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso", "u_neck", "head", ".*_humerus", ".*_mx106_block_hip"]

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)


@configclass
class SigmabanFootstepEnvCfg_PLAY(SigmabanFootstepEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

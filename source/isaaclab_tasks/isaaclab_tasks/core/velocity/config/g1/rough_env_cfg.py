# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree G1 velocity-tracking environment on rough terrain."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_VELOCITY_CFG

from ... import mdp
from ...velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            )
        },
    )
    pelvis_height = RewTerm(
        func=mdp.pelvis_height_deficit_l2,
        weight=-10.0,
        params={
            "target_height": 0.75,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough velocity tracking with 29 body actions and passive finger joints."""

    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        super().__post_init__()

        # physics
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 300
        # scene
        self.scene.robot = G1_29DOF_VELOCITY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # Body-only actions and joint observations retain the articulation's joint order.
        self.actions.joint_pos.scale = {
            ".*_hip_yaw_joint": 0.1467,
            ".*_hip_roll_joint": 0.1467,
            ".*_hip_pitch_joint": 0.11,
            ".*_knee_joint": 0.1738,
            "waist_yaw_joint": 0.11,
            "waist_roll_joint": 0.0625,
            "waist_pitch_joint": 0.0625,
            ".*_ankle_pitch_joint": 0.625,
            ".*_ankle_roll_joint": 0.625,
            ".*_shoulder_.*_joint": 0.1563,
            ".*_elbow_joint": 0.1563,
            ".*_wrist_roll_joint": 0.1563,
            ".*_wrist_pitch_joint": 0.0313,
            ".*_wrist_yaw_joint": 0.0313,
        }
        self.actions.joint_pos.joint_names = list(self.actions.joint_pos.scale)
        for term in ("joint_pos", "joint_vel"):
            getattr(self.observations.policy, term).params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=list(self.actions.joint_pos.joint_names)
            )
        # commands
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.75)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        # Enforce terrain-relative standing height after 500 PPO rollout iterations.
        self.terminations.base_height = DoneTerm(
            func=mdp.pelvis_below_terrain_clearance_after_warmup,
            params={
                "minimum_height": 0.4,
                "warmup_steps": 12_000,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # events
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

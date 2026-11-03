# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree H2 velocity-tracking environment on rough terrain."""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass, replace

from isaaclab_assets import H2_CFG

from ... import mdp
from ...velocity_env_cfg import CurriculumCfg, LocomotionVelocityRoughEnvCfg, RewardsCfg


@configclass
class H2Rewards(RewardsCfg):
    """Velocity tracking and biped locomotion rewards for H2."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    undesired_contacts = None
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_pitch_link"),
            "threshold": 0.4,
            "overstay_scale": 4.0,
        },
    )
    feet_flight = RewTerm(
        func=mdp.feet_flight,
        weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_pitch_link")},
    )
    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_pitch_link"),
        },
    )
    torso_ang_vel_xy = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    feet_lateral_distance = RewTerm(
        func=mdp.feet_lateral_distance_l2,
        weight=-16.0,
        params={
            "target_distance": 0.25,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_pitch_link", "right_ankle_pitch_link"], preserve_order=True
            ),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_pitch_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_pitch_link"),
        },
    )
    stance_knee_flexion = RewTerm(
        func=mdp.stance_knee_extension_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "max_flexion": 0.35,
            "contact_delay": 0.08,
            "terrain_radius": 0.45,
            "relief_range": (0.02, 0.12),
            "standing_scale": 0.25,
            "standing_max_flexion": 0.1,
            "terrain_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_knee_joint", "right_knee_joint"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_pitch_link", "right_ankle_pitch_link"], preserve_order=True
            ),
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_wrist_.*"]
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="head_.*")},
    )
    torso_orientation = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    shoulder_pitch_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.06,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_pitch_joint")},
    )
    knee_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_knee_joint", "right_knee_joint"], preserve_order=True
            )
        },
    )
    elbow_reference = RewTerm(
        func=mdp.joint_pos_target_l1,
        weight=-0.1,
        params={
            "target": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_elbow_joint", "right_elbow_joint"], preserve_order=True
            ),
        },
    )
    arm_lateral_posture = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"])},
    )


@configclass
class H2CurriculumCfg(CurriculumCfg):
    """Introduce stance posture gradually while retaining the terrain curriculum."""

    stance_knee = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.stance_knee_flexion.weight",
            "modify_fn": mdp.set_value_linearly,
            "modify_params": {"initial_value": 0.0, "final_value": -1.5, "start_step": 24000, "end_step": 72000},
        },
    )


@configclass
class H2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Train all 31 H2 joints to track base velocity commands over generated rough terrain."""

    rewards: H2Rewards = H2Rewards()
    curriculum: H2CurriculumCfg = H2CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = replace(H2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        for terrain in self.scene.terrain.terrain_generator.sub_terrains.values():
            terrain.convert_to_heightfield = False
        # actions
        self.actions.joint_pos.scale = 0.25
        # commands
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis", "torso_link"]
        # Limbs can support a fallen H2 without pelvis or torso contact.
        self.terminations.fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(45.0)})
        # events
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

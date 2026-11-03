# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree H2 velocity-tracking environment on flat terrain."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import mdp
from .rough_env_cfg import H2CurriculumCfg, H2Rewards, H2RoughEnvCfg


@configclass
class H2FlatRewards(H2Rewards):
    """Velocity tracking and biped locomotion rewards for H2 on flat terrain."""

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_pitch_link"),
            "threshold": 0.4,
            "overstay_scale": 12.0,
        },
    )
    feet_flight = RewTerm(
        func=mdp.feet_flight,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_pitch_link")},
    )
    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_pitch_link"),
        },
    )
    feet_lateral_distance = RewTerm(
        func=mdp.feet_lateral_distance_l2,
        weight=-1.0,
        params={
            "target_distance": 0.25,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_pitch_link", "right_ankle_pitch_link"], preserve_order=True
            ),
        },
    )
    torso_ang_vel_xy = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
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
            "terrain_cfg": None,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_knee_joint", "right_knee_joint"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_pitch_link", "right_ankle_pitch_link"], preserve_order=True
            ),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
    )
    torso_orientation = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    shoulder_pitch_deviation = None
    shoulder_pitch_posture = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_pitch_joint")},
    )
    knee_pos_limits = None
    elbow_reference = RewTerm(
        func=mdp.joint_pos_target_l1,
        weight=-0.2,
        params={
            "target": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_elbow_joint", "right_elbow_joint"], preserve_order=True
            ),
        },
    )
    walking_momentum = RewTerm(
        func=mdp.walking_yaw_momentum_l2,
        weight=0.0,
        params={"command_name": "base_velocity", "turn_std": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class H2FlatCurriculumCfg(H2CurriculumCfg):
    """Introduce stance posture and upper-body coordination penalties gradually."""

    terrain_levels = None
    stance_knee = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.stance_knee_flexion.weight",
            "modify_fn": mdp.set_value_linearly,
            "modify_params": {"initial_value": 0.0, "final_value": -1.0, "start_step": 24000, "end_step": 72000},
        },
    )
    torso_sway = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.torso_ang_vel_xy.weight",
            "modify_fn": mdp.set_value_linearly,
            "modify_params": {"initial_value": 0.0, "final_value": -0.15, "start_step": 12000, "end_step": 48000},
        },
    )
    walking_momentum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.walking_momentum.weight",
            "modify_fn": mdp.set_value_linearly,
            "modify_params": {"initial_value": 0.0, "final_value": -0.002, "start_step": 12000, "end_step": 48000},
        },
    )


@configclass
class H2FlatEnvCfg(H2RoughEnvCfg):
    """Train H2 on a plane without terrain observations or a terrain curriculum."""

    rewards: H2FlatRewards = H2FlatRewards()
    curriculum: H2FlatCurriculumCfg = H2FlatCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # physics
        self.sim.physics.newton_mjwarp.num_substeps = 8
        self.sim.physics.default = self.sim.physics.newton_mjwarp
        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # commands
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.2
        # This height check uses world coordinates and is only valid on a plane.
        self.terminations.base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.65})

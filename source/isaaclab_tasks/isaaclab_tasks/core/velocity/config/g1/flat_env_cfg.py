# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree G1 velocity-tracking environment on flat terrain."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets import G1_MINIMAL_CFG

from ... import mdp
from ...velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from .rough_env_cfg import G1EventsCfg, G1ObservationsCfg, G1Rewards


@configclass
class G1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the Unitree G1 velocity-tracking environment on flat terrain."""

    events: G1EventsCfg = G1EventsCfg()
    observations: G1ObservationsCfg = G1ObservationsCfg(height_scan=None)
    rewards: G1Rewards = G1Rewards(
        track_lin_vel_xy_exp=RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        track_ang_vel_z_exp=RewTerm(
            func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
        ),
        lin_vel_z_l2=RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2),
        dof_acc_l2=RewTerm(
            func=mdp.joint_acc_l2,
            weight=-1.0e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
        ),
        dof_torques_l2=RewTerm(
            func=mdp.joint_torques_l2,
            weight=-2.0e-6,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
        ),
        feet_air_time=RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=0.75,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": 0.4,
            },
        ),
        feet_slide=RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        ),
        joint_deviation_hip=RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
        ),
        joint_deviation_arms=RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_pitch_joint",
                        ".*_elbow_roll_joint",
                    ],
                )
            },
        ),
        joint_deviation_fingers=RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_five_joint",
                        ".*_three_joint",
                        ".*_six_joint",
                        ".*_four_joint",
                        ".*_zero_joint",
                        ".*_one_joint",
                        ".*_two_joint",
                    ],
                )
            },
        ),
        joint_deviation_torso=RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
        ),
        pelvis_height=None,
        torso_ang_vel=None,
        root_downward_speed=None,
        feet_support_edge=None,
    )

    def __post_init__(self):
        super().__post_init__()

        # physics
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 95
        newton_mjwarp.solver_cfg.nconmax = 10
        self.sim.physics.default = newton_mjwarp
        # scene
        self.scene.robot = replace(G1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # commands
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.75)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # curriculum
        self.curriculum.terrain_levels = None

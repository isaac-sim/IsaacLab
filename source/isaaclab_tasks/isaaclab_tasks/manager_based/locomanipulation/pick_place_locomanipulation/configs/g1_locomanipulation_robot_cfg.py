# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 locomanipulation robot.

This module provides configurations for the Unitree G1 humanoid robot specifically
designed for locomanipulation tasks, including both fixed base and mobile
configurations for upper body manipulation.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration for G1 Locomanipulation
##

G1_LOCOMANIPULATION_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            "omniverse://isaac-dev.ov.nvidia.com/Projects/agile/Robots/Collected_g1/g1_minimal_with_leg_collision.usd"
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,  # Configurable - can be set to True for fixed base
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(0.7071, 0, 0, 0.7071),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
            saturation_effort=180.0,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.01,
            saturation_effort=80.0,
        ),
        "waist": DCMotorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.01,
            saturation_effort=120.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
                ".*_hand_.*",
            ],
            effort_limit=300,
            velocity_limit=100,
            stiffness={
                ".*_shoulder_.*": 4000,
                ".*_elbow_.*": 4000,
                ".*_wrist_.*": 4000,
                ".*_hand_.*": 10,
            },
            damping={
                ".*_shoulder_.*": 50,
                ".*_elbow_.*": 50,
                ".*_wrist_.*": 50,
                ".*_hand_.*": 0.2,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*_joint": 0.01,
                ".*_hand_.*": 0.001,
            },
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_.*",
            ],
            effort_limit={
                ".*_hand_.*": 2.0,
            },
            velocity_limit={
                ".*_hand_.*": 10.0,
            },
            stiffness={
                ".*_hand_.*": 10.0,
            },
            damping={
                ".*_hand_.*": 0.2,
            },
            armature={
                ".*_hand_.*": 0.01,
            },
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)
"""Configuration for the Unitree G1 Humanoid robot for locomanipulation tasks.

This configuration sets up the G1 humanoid robot for locomanipulation tasks,
allowing both locomotion and manipulation capabilities. The robot can be configured
for either fixed base or mobile scenarios by modifying the fix_root_link parameter.

Key features:
- Configurable base (fixed or mobile) via fix_root_link parameter

Usage examples:
    # For fixed base scenarios (upper body manipulation only)
    fixed_base_cfg = G1_LOCOMANIPULATION_ROBOT_CFG.copy()
    fixed_base_cfg.spawn.articulation_props.fix_root_link = True

    # For mobile scenarios (locomotion + manipulation)
    mobile_cfg = G1_LOCOMANIPULATION_ROBOT_CFG.copy()
    mobile_cfg.spawn.articulation_props.fix_root_link = False
"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the Valkyrie robot from IHMC Robotics.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

spawn_usd = sim_utils.UsdFileCfg(
    usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Valkyrie/valkyrie.usd",
    activate_contact_sensors=True,
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
        enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    ),
)

VALKYRIE_CFG = ArticulationCfg(
    spawn=spawn_usd,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.2),
        joint_pos={
            ".*HipPitch": -0.20,
            ".*KneePitch": 0.42,
            ".*AnklePitch": -0.23,
            "leftShoulderPitch": 0.27,
            "rightShoulderPitch": 0.27,
            "leftElbowPitch": -0.7,
            "rightElbowPitch": 0.7,
            "leftShoulderRoll": -1.2,
            "rightShoulderRoll": 1.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "torso": ImplicitActuatorCfg(
            effort_limit=2000,
            joint_names_expr=["torso.*"],
            stiffness={
                "torsoPitch": 150.0,
                "torsoRoll": 150.0,
                "torsoYaw": 150.0,
            },
            damping={
                "torsoPitch": 5.0,
                "torsoRoll": 5.0,
                "torsoYaw": 5.0,
            },
            armature=0.01,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*HipYaw",
                ".*HipRoll",
                ".*HipPitch",
                ".*Knee.*",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*HipYaw": 150.0,
                ".*HipRoll": 150.0,
                ".*HipPitch": 200.0,
                ".*Knee.*": 200.0,
            },
            damping={
                ".*HipYaw": 5.0,
                ".*HipRoll": 5.0,
                ".*HipPitch": 5.0,
                ".*Knee.*": 5.0,
            },
            armature={
                ".*Hip.*": 0.01,
                ".*Knee.*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*AnklePitch", ".*AnkleRoll"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Shoulder.*",
                ".*Elbow.*",
                ".*Forearm.*",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*Shoulder.*": 0.01,
                ".*Elbow.*": 0.01,
                ".*Forearm.*": 0.001,
            },
        ),
    },
)

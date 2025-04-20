# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the NAO robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

NAO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"C:/Users/reill/IsaacLab/nao/nao2/nao_nohands.usd",  # Path to NAO USD file
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,  # Lower for smaller robot
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,  # Increased for better stability
            sleep_threshold=0.005,  # Lowered for more responsive behavior
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.345),  # NAO is about 58cm tall, starting with feet on ground
        joint_pos={
            # Define specific initial poses for better stability if needed
            # Example: starting with slightly bent knees
            # ".*KneePitch": 0.1,
            "HeadYaw": 0.0,
            "HeadPitch": 0.0,
            
            # Arms
            ".*ShoulderPitch": 1.4,
            "RShoulderRoll": -0.05,
            "LShoulderRoll": 0.05,
            ".*ElbowYaw": 0.0,
            ".*WristYaw": 0.0,
            # ".*Hand": 1.0,
            
            # Legs
            ".*HipYawPitch": 0.0,
            ".*HipRoll": 0.0,
            ".*HipPitch": 0.0,
            ".*KneePitch": 0.0,
            ".*AnklePitch": 0.0,
            ".*AnkleRoll": 0.0,
            "RElbowRoll": 0.05,
            "LElbowRoll": -0.05,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # Target all joints
            stiffness={
                # Head
                "HeadYaw": 5.0,
                "HeadPitch": 5.0,
                
                # Arms
                ".*ShoulderPitch": 8.0,
                ".*ShoulderRoll": 8.0,
                ".*ElbowYaw": 7.0,
                ".*ElbowRoll": 7.0,
                ".*WristYaw": 3.0,
                # ".*Hand": 1.0,
                
                # Legs
                ".*HipYawPitch": 15.0,
                ".*HipRoll": 15.0,
                ".*HipPitch": 15.0,
                ".*KneePitch": 10.0,
                ".*AnklePitch": 8.0,
                ".*AnkleRoll": 8.0,
            },
            damping={
                # Head
                "HeadYaw": 1.0,
                "HeadPitch": 1.0,
                
                # Arms
                ".*ShoulderPitch": 2.0,
                ".*ShoulderRoll": 2.0,
                ".*ElbowYaw": 1.5,
                ".*ElbowRoll": 1.5,
                ".*WristYaw": 0.8,
                # ".*Hand": 0.5,
                
                # Legs
                ".*HipYawPitch": 4.0,
                ".*HipRoll": 4.0,
                ".*HipPitch": 4.0,
                ".*KneePitch": 3.0,
                ".*AnklePitch": 2.0,
                ".*AnkleRoll": 2.0,
            },
        ),
    },
)
"""Configuration for the NAO robot."""

# You can keep the HUMANOID_CFG if needed or comment it out
"""
HUMANOID_CFG = ArticulationCfg(
    # Original humanoid configuration...
)
"""
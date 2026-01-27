# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the T3 Hexapod robot.

T3 is a 6-legged robot with 12 joints (2 joints per leg: Leg + Calf).

Leg naming convention:
- FL: Front Left
- ML: Middle Left
- RL: Rear Left
- FR: Front Right
- MR: Middle Right
- RR: Rear Right

Joint naming:
- *_Leg_joint: Hip joint (connects body to upper leg)
- *_Calf_joint: Knee joint (connects upper leg to lower leg)
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Actuators
##

T3_SIMPLE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*_Leg_joint", ".*_Calf_joint"],
    effort_limit=20.0,
    velocity_limit=10.0,
    stiffness={".*_Leg_joint": 25.0, ".*_Calf_joint": 25.0},
    damping={".*_Leg_joint": 0.5, ".*_Calf_joint": 0.5},
)
"""Configuration for T3 hexapod with implicit actuator model."""


T3_DC_MOTOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_Leg_joint", ".*_Calf_joint"],
    saturation_effort=30.0,
    effort_limit=20.0,
    velocity_limit=10.0,
    stiffness={".*_Leg_joint": 25.0, ".*_Calf_joint": 25.0},
    damping={".*_Leg_joint": 0.5, ".*_Calf_joint": 0.5},
)
"""Configuration for T3 hexapod with DC motor actuator model."""


##
# Configuration - Articulation (using URDF)
##

# Path to URDF file - you need to convert this to USD first
# Or use UrdfFileCfg to load directly from URDF
T3_URDF_PATH = "source/isaaclab_assets/data/Robots/T3/t3.urdf"


T3_HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=T3_URDF_PATH,
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # URDF specific settings
        fix_base=False,  # Floating base for locomotion
        merge_fixed_joints=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),  # Start position above ground
        rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion (w, x, y, z)
        joint_pos={
            # Left side legs - leg joints
            "FL_Leg_joint": 0.0,
            "ML_Leg_joint": 0.0,
            "RL_Leg_joint": 0.0,
            # Right side legs - leg joints
            "FR_Leg_joint": 0.0,
            "MR_Leg_joint": 0.0,
            "RR_Leg_joint": 0.0,
            # All calf joints (bent)
            "FL_Calf_joint": -0.8,
            "ML_Calf_joint": -0.8,
            "RL_Calf_joint": -0.8,
            "FR_Calf_joint": -0.8,
            "MR_Calf_joint": -0.8,
            "RR_Calf_joint": -0.8,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"legs": T3_SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for T3 Hexapod robot using URDF file with implicit actuator."""


# Alternative configuration using DC Motor model
T3_HEXAPOD_DC_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=T3_URDF_PATH,
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        fix_base=False,
        merge_fixed_joints=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "FL_Leg_joint": 0.0,
            "ML_Leg_joint": 0.0,
            "RL_Leg_joint": 0.0,
            "FR_Leg_joint": 0.0,
            "MR_Leg_joint": 0.0,
            "RR_Leg_joint": 0.0,
            "FL_Calf_joint": -0.8,
            "ML_Calf_joint": -0.8,
            "RL_Calf_joint": -0.8,
            "FR_Calf_joint": -0.8,
            "MR_Calf_joint": -0.8,
            "RR_Calf_joint": -0.8,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"legs": T3_DC_MOTOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for T3 Hexapod robot with DC motor actuator model."""

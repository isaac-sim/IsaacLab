# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Disney Research robots.

The following configuration parameters are available:

* :obj:`BDX_CFG`: The BD-X robot with implicit Actuator model

Reference:

* https://github.com/rimim/AWD/tree/main/awd/data/assets/go_bdx

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

BDX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Disney/BDX/bdx.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", ".*_ankle"],
            stiffness={
                ".*_hip_yaw": 100.0,
                ".*_hip_roll": 80.0,
                ".*_hip_pitch": 120.0,
                ".*_knee": 200.0,
                ".*_ankle": 200.0,
            },
            damping={
                ".*_hip_yaw": 3.0,
                ".*_hip_roll": 3.0,
                ".*_hip_pitch": 6.0,
                ".*_knee": 6.0,
                ".*_ankle": 6.0,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["neck_pitch", "head_pitch", "head_yaw", "head_roll", ".*_antenna"],
            stiffness={
                ".*": 20.0,
            },
            damping={
                ".*": 1.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for the Disney BD-X robot with implicit actuator model."""

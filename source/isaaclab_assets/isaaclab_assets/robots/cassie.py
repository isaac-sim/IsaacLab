# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Agility robots.

The following configurations are available:

* :obj:`CASSIE_CFG`: Agility Cassie robot with simple PD controller for the legs

Reference: https://github.com/UMich-BipedLab/Cassie_Model/blob/master/urdf/cassie.urdf
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

CASSIE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Agility/Cassie/cassie.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            "hip_abduction_left": 0.1,
            "hip_rotation_left": 0.0,
            "hip_flexion_left": 1.0,
            "thigh_joint_left": -1.8,
            "ankle_joint_left": 1.57,
            "toe_joint_left": -1.57,
            "hip_abduction_right": -0.1,
            "hip_rotation_right": 0.0,
            "hip_flexion_right": 1.0,
            "thigh_joint_right": -1.8,
            "ankle_joint_right": 1.57,
            "toe_joint_right": -1.57,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["hip_.*", "thigh_.*", "ankle_.*"],
            control_mode="position",
            effort_limit_sim=200.0,
            stiffness={
                "hip_abduction.*": 100.0,
                "hip_rotation.*": 100.0,
                "hip_flexion.*": 200.0,
                "thigh_joint.*": 200.0,
                "ankle_joint.*": 200.0,
            },
            damping={
                "hip_abduction.*": 3.0,
                "hip_rotation.*": 3.0,
                "hip_flexion.*": 6.0,
                "thigh_joint.*": 6.0,
                "ankle_joint.*": 6.0,
            },
            armature=1e-3,
            friction=1e-5,
        ),
        "toes": ImplicitActuatorCfg(
            joint_names_expr=["toe_.*"],
            control_mode="position",
            effort_limit_sim=20.0,
            stiffness={
                "toe_joint.*": 20.0,
            },
            damping={
                "toe_joint.*": 1.0,
            },
            armature=1e-3,
            friction=1e-5,
        ),
    },
)

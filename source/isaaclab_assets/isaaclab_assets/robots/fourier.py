# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Fourier Robots.

The following configuration parameters are available:

* :obj:`GR1T2_CFG`: The GR1T2 humanoid.

Reference: https://www.fftai.com/products-gr1
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##


GR1T2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "trunk": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_.*",
                ".*_knee_.*",
                ".*_ankle_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "right-arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_.*",
                "right_elbow_.*",
                "right_wrist_.*",
            ],
            effort_limit=torch.inf,
            velocity_limit=torch.inf,
            stiffness=None,
            damping=None,
            armature=0.0,
        ),
        "left-arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_.*",
                "left_elbow_.*",
                "left_wrist_.*",
            ],
            effort_limit=torch.inf,
            velocity_limit=torch.inf,
            stiffness=None,
            damping=None,
            armature=0.0,
        ),
        "right-hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "R_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "left-hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
    },
)
"""Configuration for the GR1T2 Humanoid robot."""

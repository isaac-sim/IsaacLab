# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rethink Robotics arms.

The following configuration parameters are available:

* :obj:`SAWYER_CFG`: The Sawyer arm without any tool attached.

Reference: https://github.com/RethinkRobotics/sawyer_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

SAWYER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/RethinkRobotics/Sawyer/sawyer_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "head_pan": 0.0,
            "right_j0": 0.0,
            "right_j1": -0.785,
            "right_j2": 0.0,
            "right_j3": 1.05,
            "right_j4": 0.0,
            "right_j5": 1.3,
            "right_j6": 0.0,
        },
    ),
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan"],
            effort_limit_sim=8.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["right_j[0-6]"],
            effort_limit_sim={
                "right_j[0-1]": 80.0,
                "right_j[2-3]": 40.0,
                "right_j[4-6]": 9.0,
            },
            stiffness=100.0,
            damping=4.0,
        ),
    },
)
"""Configuration of Rethink Robotics Sawyer arm."""

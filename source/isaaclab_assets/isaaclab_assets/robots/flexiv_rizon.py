# Copyright (c) 2026-2027, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Flexiv Rizon robots.

The following configurations are available:

* :obj:`FLEXIV_RIZON4S_CFG`: The Flexiv Rizon 4s arm without a gripper.

Reference: https://www.flexiv.com/product/rizon
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

FLEXIV_RIZON4S_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Flexiv/Rizon4s/rizon4s.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.785,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 1.571,
            "joint7": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        # Joints 1-2: Higher torque (123 Nm), lower speed (120°/s = 2.094 rad/s)
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=123.0,
            velocity_limit_sim=2.094,
            stiffness=None,
            damping=None,
        ),
        # Joints 3-4: Medium torque (64 Nm), medium speed (140°/s = 2.443 rad/s)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-4]"],
            effort_limit_sim=64.0,
            velocity_limit_sim=2.443,
            stiffness=None,
            damping=None,
        ),
        # Joints 5-7: Lower torque (39 Nm), higher speed (280°/s = 4.887 rad/s)
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            effort_limit_sim=39.0,
            velocity_limit_sim=4.887,
            stiffness=None,
            damping=None,
        ),
    },
)

"""Configuration of Flexiv Rizon 4s arm using implicit actuator models."""

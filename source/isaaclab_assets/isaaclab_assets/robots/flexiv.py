# Copyright (c) 2026-2027, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Flexiv Rizon robots.

The following configurations are available:

* :obj:`FLEXIV_RIZON4S_CFG`: The Flexiv Rizon 4s arm without a gripper.
* :obj:`FLEXIV_RIZON4S_GRAV_GRIPPER_CFG`: The Flexiv Rizon 4s arm with Grav gripper.

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
            disable_gravity=True,
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
            "joint2": -0.698,
            "joint3": 0.0,
            "joint4": 1.571,
            "joint5": 0.0,
            "joint6": 0.698,
            "joint7": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=123.0,
            velocity_limit_sim=2.094,
            stiffness=6000.0,
            damping=108.5,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-4]"],
            effort_limit_sim=64.0,
            velocity_limit_sim=2.443,
            stiffness=4200.0,
            damping=90.7,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            effort_limit_sim=39.0,
            velocity_limit_sim=4.887,
            stiffness=1500.0,
            damping=54.2,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""Configuration of Flexiv Rizon 4s arm using implicit actuator models."""


FLEXIV_RIZON4S_GRAV_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Flexiv/Rizon4s/rizon4s_with_grav.usd",
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
            "joint2": -0.698,
            "joint3": 0.0,
            "joint4": 1.571,
            "joint5": 0.0,
            "joint6": 0.698,
            "joint7": 0.0,
            "finger_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "right_outer_finger_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=123.0,
            velocity_limit_sim=2.094,
            stiffness=1320.0,
            damping=72.0,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-4]"],
            effort_limit_sim=64.0,
            velocity_limit_sim=2.443,
            stiffness=600.0,
            damping=35.0,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            effort_limit_sim=39.0,
            velocity_limit_sim=4.887,
            stiffness=216.0,
            damping=29.0,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.6,
            stiffness=2e3,
            damping=1e1,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""Configuration of Flexiv Rizon 4s arm with Grav gripper using implicit actuator models.

The Grav gripper is a parallel gripper with the following joint configuration:
- finger_joint: Main actuation joint (opened: 45 deg, closed: -8.88 deg)
- *_knuckle_joint: Passive/mimic joints (not directly actuated)

End effector body: right_finger_tip
"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Galbot humanoid robot.

The following configuration parameters are available:

* :obj:`GALBOT_ONE_CHARLIE_CFG`: The galbot_one_charlie humanoid robot.

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##


GALBOT_ONE_CHARLIE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Galbot/galbot_one_charlie/galbot_one_charlie.usd",
        variants={"Physics": "PhysX"},
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "leg_joint1": 0.8,
            "leg_joint2": 2.3,
            "leg_joint3": 1.55,
            "leg_joint4": 0.0,
            "head_joint1": 0.0,
            "head_joint2": 0.36,
            "left_arm_joint1": -0.5480,
            "left_arm_joint2": -0.6551,
            "left_arm_joint3": 2.407,
            "left_arm_joint4": 1.3641,
            "left_arm_joint5": -0.4416,
            "left_arm_joint6": 0.1168,
            "left_arm_joint7": 1.2308,
            "left_gripper_left_joint": 0.035,
            "left_gripper_right_joint": 0.035,
            "right_arm_joint1": 0.1535,
            "right_arm_joint2": 1.0087,
            "right_arm_joint3": 0.0895,
            "right_arm_joint4": 1.5743,
            "right_arm_joint5": -0.2422,
            "right_arm_joint6": -0.0009,
            "right_arm_joint7": -0.9143,
            "right_suction_cup_joint1": 0.0,
        },
        pos=(-0.6, 0.0, -0.8),
    ),
    # PD parameters are read from USD file with Gain Tuner
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "leg": ImplicitActuatorCfg(
            joint_names_expr=["leg_joint.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint.*", "right_suction_cup_joint1"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper_.*_joint"],
            velocity_limit_sim=1.0,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
    },
)
"""Configuration of Galbot_one_charlie humanoid using implicit actuator models."""

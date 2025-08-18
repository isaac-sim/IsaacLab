# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Galbot humanoid robot.

The following configuration parameters are available:

* :obj:`GALBOT_ONE_CHARLIE_CFG`: The galbot_one_charlie humanoid robot.

"""

import os

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


##
# Configuration
##


GALBOT_ONE_CHARLIE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Galbot/galbot_one_charlie.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "right_arm_joint1": 0.1535,
            "right_arm_joint2": 1.0087,
            "right_arm_joint3": 0.0895,
            "right_arm_joint4": 1.5743,
            "right_arm_joint5": -0.2422,
            "right_arm_joint6": -0.0009,
            "right_arm_joint7": -0.9143,
            "leg_joint1": 0.8,
            "leg_joint2": 2.3,
            "leg_joint3": 1.55,
            "leg_joint4": 0.0,
            "head_joint1": 0.0,
            "left_arm_joint1": -0.5480,
            "head_joint2": 0.36,
            "left_arm_joint2": -0.6551,
            "left_arm_joint3": 2.407,
            "left_arm_joint4": 1.3641,
            "left_arm_joint5": -0.4416,
            "left_arm_joint6": 0.1168,
            "left_arm_joint7": 1.2308,
            "left_gripper_left_joint": 0.035,
            "left_gripper_right_joint": 0.035,
            "right_suction_cup_joint1": 0.0,
        },
        pos=(-0.6, 0.0, -0.8),
    ),
    # PD parameters are read from USD file with Gain Tuner
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint.*"],
            velocity_limit_sim=1.50,
            effort_limit_sim=4.0,
            stiffness=17453293764608.0,
            damping=1745.32922,
        ),
        "leg": ImplicitActuatorCfg(
            joint_names_expr=["leg_joint.*"],
            velocity_limit_sim=1.0,
            effort_limit_sim={"leg_joint[1,2]": 507.0, "leg_joint[3]": 252.0, "leg_joint[4]": 90.0},
            stiffness={"leg_joint[1,3]": 17453293764608.0, "leg_joint[2,4]": 17453.292972},
            damping={"leg_joint[1-3]": 1745.32922, "leg_joint[4]": 17453.29297},
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint.*"],
            velocity_limit_sim=1.5,
            effort_limit_sim={"left_arm_joint[1-2]": 60.0, "left_arm_joint[3-4]": 30.0, "left_arm_joint[5-7]": 10.0},
            stiffness={
                "left_arm_joint[1,3,4,6]": 1745.32922 * 1e3,
                "left_arm_joint2": 174.53293 * 1e3,
                "left_arm_joint[5,7]": 1745329217077248.0,
            },
            damping=1745.32922,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint.*", "right_suction_cup_joint1"],
            velocity_limit_sim=1.5,
            effort_limit_sim={
                "right_arm_joint[1-2]": 60.0,
                "right_arm_joint[3-4]": 30.0,
                "right_arm_joint[5-7]": 10.0,
                "right_suction_cup_joint1": 1.0,
            },
            stiffness={
                "right_arm_joint[1,5]": 1745.32922,
                "right_arm_joint[2,6,7]": 1745329217077248.0,
                "right_arm_joint[3]": 17453.29297,
                "right_arm_joint[4]": 17453293764608.0,
                "right_suction_cup_joint1": 1745.32922,
            },
            damping={"right_arm_joint[1-7]": 1745.32922, "right_suction_cup_joint1": 1745.32922},
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper_.*_joint"],
            velocity_limit_sim=2.0,
            effort_limit_sim=15.0,
            stiffness=10000000000.0,
            damping=100000.0,
        ),
    },
)
"""Configuration of Galbot_one_charlie humanoid using implicit actuator models."""

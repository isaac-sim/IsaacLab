# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka R800 robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
##
# Configuration
##

KUKA_VICTOR_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/victor/victor_left_arm_with_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        )
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        joint_pos={
            # arm states
            "victor_left_arm_joint_1": 1.3661269501533881,
            "victor_left_arm_joint_2": -0.5341374194622199,
            "victor_left_arm_joint_3": 2.383251686578518,
            "victor_left_arm_joint_4": 1.6179420456098288,
            "victor_left_arm_joint_5": -2.204557118713759,
            "victor_left_arm_joint_6": 1.1547660552023602,
            "victor_left_arm_joint_7": 0.5469460457579646,
            # gripper finger states
            "victor_left_finger_a_joint_1": 0.890168571428571,
            "victor_left_finger_a_joint_2": 0,
            "victor_left_finger_a_joint_3": -0.8901685714285714,
            "victor_left_finger_b_joint_1": 0.890168571428571,
            "victor_left_finger_b_joint_2": 0,
            "victor_left_finger_b_joint_3": -0.8901685714285714,
            "victor_left_finger_c_joint_1": 0.890168571428571,
            "victor_left_finger_c_joint_2": 0,
            "victor_left_finger_c_joint_3": -0.8901685714285714,
            # gripper scissors states
            "victor_left_palm_finger_b_joint": 0.115940392156862,
            "victor_left_palm_finger_c_joint": -0.11594039215686275,
        },
    ),
    actuators={
        "victor_left_arm": ImplicitActuatorCfg(
            joint_names_expr=["victor_left_arm_joint.*"],
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "victor_left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["victor_left.*finger.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Kuka iiwa robot."""

KUKA_VICTOR_LEFT_HIGH_PD_CFG = KUKA_VICTOR_LEFT_CFG.copy()
KUKA_VICTOR_LEFT_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
KUKA_VICTOR_LEFT_HIGH_PD_CFG.actuators["victor_left_arm"].stiffness = 400.0
KUKA_VICTOR_LEFT_HIGH_PD_CFG.actuators["victor_left_arm"].damping = 80.0
"""Configuration of Kuka iiwa with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""


KUKA_VICTOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/zixuanh/force/Isaacgym_Env/urdf/victor/robots/victor_approx_gripper/victor_approx_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        )
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(-0.4, -0.35, -0.8),
        pos=(0, 0, 0),
        joint_pos={
            # arm states
            # "victor_left_arm_joint_1": -0.694,
            # "victor_left_arm_joint_2": 0.140,
            # "victor_left_arm_joint_3": -0.229,
            # "victor_left_arm_joint_4": -1.110,
            # "victor_left_arm_joint_5": -0.512,
            # "victor_left_arm_joint_6": 1.272,
            # "victor_left_arm_joint_7": 0.077,
            # "victor_right_arm_joint_1": 0.724,
            # "victor_right_arm_joint_2": 0.451,
            # "victor_right_arm_joint_3": 0.940,
            # "victor_right_arm_joint_4": -1.425,
            # "victor_right_arm_joint_5": 0.472,
            # "victor_right_arm_joint_6": 0.777,
            # "victor_right_arm_joint_7": -0.809,
            
            "victor_left_arm_joint_1": 1.3661269501533881,
            "victor_left_arm_joint_2": -0.5341374194622199,
            "victor_left_arm_joint_3": 2.383251686578518,
            "victor_left_arm_joint_4": 1.6179420456098288,
            "victor_left_arm_joint_5": -2.204557118713759,
            "victor_left_arm_joint_6": 1.1547660552023602,
            "victor_left_arm_joint_7": 0.5469460457579646,
            "victor_right_arm_joint_1": 0.724,
            "victor_right_arm_joint_2": 0.451,
            "victor_right_arm_joint_3": 0.940,
            "victor_right_arm_joint_4": -1.425,
            "victor_right_arm_joint_5": 0.472,
            "victor_right_arm_joint_6": 0.777,
            "victor_right_arm_joint_7": -0.809,
            
            # gripper finger states
            "victor_left_finger_a_joint_1": 0.890168571428571,
            "victor_left_finger_a_joint_2": 0,
            "victor_left_finger_a_joint_3": -0.8901685714285714,
            "victor_left_finger_b_joint_1": 0.890168571428571,
            "victor_left_finger_b_joint_2": 0,
            "victor_left_finger_b_joint_3": -0.8901685714285714,
            "victor_left_finger_c_joint_1": 0.890168571428571,
            "victor_left_finger_c_joint_2": 0,
            "victor_left_finger_c_joint_3": -0.8901685714285714,
            
            "victor_right_finger_a_joint_1": 0.890168571428571,
            "victor_right_finger_a_joint_2": 0,
            "victor_right_finger_a_joint_3": -0.8901685714285714,
            "victor_right_finger_b_joint_1": 0.890168571428571,
            "victor_right_finger_b_joint_2": 0,
            "victor_right_finger_b_joint_3": -0.8901685714285714,
            "victor_right_finger_c_joint_1": 0.890168571428571,
            "victor_right_finger_c_joint_2": 0,
            "victor_right_finger_c_joint_3": -0.8901685714285714,
            
            # gripper scissors states
            "victor_left_palm_finger_b_joint": 0.115940392156862,
            "victor_left_palm_finger_c_joint": -0.11594039215686275,
            "victor_right_palm_finger_b_joint": 0.115940392156862,
            "victor_right_palm_finger_c_joint": -0.11594039215686275,
        },
    ),
    actuators={
        "victor_left_arm": ImplicitActuatorCfg(
            joint_names_expr=["victor_left_arm_joint.*"],
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "victor_right_arm": ImplicitActuatorCfg(
            joint_names_expr=["victor_right_arm_joint.*"],
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "victor_left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["victor_left.*finger.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "victor_right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["victor_right.*finger.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Kuka iiwa robot."""

KUKA_VICTOR_HIGH_PD_CFG = KUKA_VICTOR_CFG.copy()
KUKA_VICTOR_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
KUKA_VICTOR_HIGH_PD_CFG.actuators["victor_left_arm"].stiffness = 400.0
KUKA_VICTOR_HIGH_PD_CFG.actuators["victor_left_arm"].damping = 80.0
KUKA_VICTOR_HIGH_PD_CFG.actuators["victor_right_arm"].stiffness = 400.0
KUKA_VICTOR_HIGH_PD_CFG.actuators["victor_right_arm"].damping = 80.0
"""Configuration of Kuka iiwa with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

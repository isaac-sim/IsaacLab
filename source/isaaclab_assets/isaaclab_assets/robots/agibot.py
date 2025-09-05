# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Agibot A2D humanoid robots.

The following configurations are available:

* :obj:`AGIBOT_A2D_CFG`: Agibot A2D robot


"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

AGIBOT_A2D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Agibot/A2D/A2D_physics.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Body joints
            "joint_lift_body": 0.1995,
            "joint_body_pitch": 0.6025,
            # Head joints
            "joint_head_yaw": 0.0,
            "joint_head_pitch": 0.6708,
            # Left arm joints
            "left_arm_joint1": -1.0817,
            "left_arm_joint2": 0.5907,
            "left_arm_joint3": 0.3442,
            "left_arm_joint4": -1.2819,
            "left_arm_joint5": 0.6928,
            "left_arm_joint6": 1.4725,
            "left_arm_joint7": -0.1599,
            # Right arm joints
            "right_arm_joint1": 1.0817,
            "right_arm_joint2": -0.5907,
            "right_arm_joint3": -0.3442,
            "right_arm_joint4": 1.2819,
            "right_arm_joint5": -0.6928,
            "right_arm_joint6": -0.7,
            "right_arm_joint7": 0.0,
            # Left gripper joints
            "left_Right_1_Joint": 0.0,
            "left_hand_joint1": 0.994,
            "left_Right_0_Joint": 0.0,
            "left_Left_0_Joint": 0.0,
            "left_Right_Support_Joint": 0.994,
            "left_Left_Support_Joint": 0.994,
            "left_Right_RevoluteJoint": 0.0,
            "left_Left_RevoluteJoint": 0.0,
            # Right gripper joints
            "right_Right_1_Joint": 0.0,
            "right_hand_joint1": 0.994,
            "right_Right_0_Joint": 0.0,
            "right_Left_0_Joint": 0.0,
            "right_Right_Support_Joint": 0.994,
            "right_Left_Support_Joint": 0.994,
            "right_Right_RevoluteJoint": 0.0,
            "right_Left_RevoluteJoint": 0.0,
        },
        pos=(-0.6, 0.0, -1.05),  # init pos of the articulation for teleop
    ),
    actuators={
        # Body lift and torso actuators
        "body": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift_body", "joint_body_pitch"],
            effort_limit_sim=10000.0,
            velocity_limit_sim=2.61,
            stiffness=10000000.0,
            damping=200.0,
        ),
        # Head actuators
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_yaw", "joint_head_pitch"],
            effort_limit_sim=50.0,
            velocity_limit_sim=1.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # Left arm actuator
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint[1-7]"],
            effort_limit_sim={
                "left_arm_joint1": 2000.0,
                "left_arm_joint[2-7]": 1000.0,
            },
            velocity_limit_sim=1.57,
            stiffness={"left_arm_joint1": 10000000.0, "left_arm_joint[2-7]": 20000.0},
            damping={"left_arm_joint1": 0.0, "left_arm_joint[2-7]": 0.0},
        ),
        # Right arm actuator
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint[1-7]"],
            effort_limit_sim={
                "right_arm_joint1": 2000.0,
                "right_arm_joint[2-7]": 1000.0,
            },
            velocity_limit_sim=1.57,
            stiffness={"right_arm_joint1": 10000000.0, "right_arm_joint[2-7]": 20000.0},
            damping={"right_arm_joint1": 0.0, "right_arm_joint[2-7]": 0.0},
        ),
        # "left_Right_2_Joint" is excluded from Articulation.
        # "left_hand_joint1" is the driver joint, and "left_Right_1_Joint" is the mimic joint.
        # "left_.*_Support_Joint" driver joint can be set optionally, to disable the driver, set stiffness and damping to 0.0 below
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_hand_joint1", "left_.*_Support_Joint"],
            effort_limit_sim={"left_hand_joint1": 10.0, "left_.*_Support_Joint": 1.0},
            velocity_limit_sim=2.0,
            stiffness={"left_hand_joint1": 20.0, "left_.*_Support_Joint": 2.0},
            damping={"left_hand_joint1": 0.10, "left_.*_Support_Joint": 0.01},
        ),
        # set PD to zero for passive joints in close-loop gripper
        "left_gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["left_.*_(0|1)_Joint", "left_.*_RevoluteJoint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # "right_Right_2_Joint" is excluded from Articulation.
        # "right_hand_joint1" is the driver joint, and "right_Right_1_Joint" is the mimic joint.
        # "right_.*_Support_Joint" driver joint can be set optionally, to disable the driver, set stiffness and damping to 0.0 below
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_joint1", "right_.*_Support_Joint"],
            effort_limit_sim={"right_hand_joint1": 100.0, "right_.*_Support_Joint": 100.0},
            velocity_limit_sim=10.0,
            stiffness={"right_hand_joint1": 20.0, "right_.*_Support_Joint": 2.0},
            damping={"right_hand_joint1": 0.10, "right_.*_Support_Joint": 0.01},
        ),
        # set PD to zero for passive joints in close-loop gripper
        "right_gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["right_.*_(0|1)_Joint", "right_.*_RevoluteJoint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

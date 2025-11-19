# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.
* :obj:`UR10E_ROBOTIQ_GRIPPER_CFG`: The UR10E arm with Robotiq_2f_140 gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

UR10e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 3.141592653589793,
            "shoulder_lift_joint": -1.5707963267948966,
            "elbow_joint": 1.5707963267948966,
            "wrist_1_joint": -1.5707963267948966,
            "wrist_2_joint": -1.5707963267948966,
            "wrist_3_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64101615,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
    },
)

"""Configuration of UR-10 arm using implicit actuator models."""

UR10_LONG_SUCTION_CFG = UR10_CFG.copy()
UR10_LONG_SUCTION_CFG.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10/ur10.usd"
UR10_LONG_SUCTION_CFG.spawn.variants = {"Gripper": "Long_Suction"}
UR10_LONG_SUCTION_CFG.spawn.rigid_props.disable_gravity = True
UR10_LONG_SUCTION_CFG.init_state.joint_pos = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5707,
    "elbow_joint": 1.5707,
    "wrist_1_joint": -1.5707,
    "wrist_2_joint": 1.5707,
    "wrist_3_joint": 0.0,
}

"""Configuration of UR10 arm with long suction gripper."""

UR10_SHORT_SUCTION_CFG = UR10_LONG_SUCTION_CFG.copy()
UR10_SHORT_SUCTION_CFG.spawn.variants = {"Gripper": "Short_Suction"}

"""Configuration of UR10 arm with short suction gripper."""

UR10e_ROBOTIQ_GRIPPER_CFG = UR10e_CFG.copy()
UR10e_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2f_140"}
UR10e_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
UR10e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos["finger_joint"] = 0.0
UR10e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_inner_finger_joint"] = 0.0
UR10e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_inner_finger_pad_joint"] = 0.0
UR10e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_outer_.*_joint"] = 0.0
# the major actuator joint for gripper
UR10e_ROBOTIQ_GRIPPER_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
)
# the auxiliary actuator joint for gripper
UR10e_ROBOTIQ_GRIPPER_CFG.actuators["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.2,
    damping=0.001,
    friction=0.0,
    armature=0.0,
)
# the passive joints for gripper
UR10e_ROBOTIQ_GRIPPER_CFG.actuators["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_pad_joint", ".*_outer_finger_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)

"""Configuration of UR-10E arm with Robotiq_2f_140 gripper."""

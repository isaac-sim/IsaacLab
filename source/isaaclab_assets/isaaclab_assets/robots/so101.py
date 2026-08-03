# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the TheRobotStudio SO-101 follower arm.

The following configurations are available:

* :obj:`SO101_CFG`: SO-101 5-DOF arm with a single-jaw gripper.
* :obj:`SO101_HIGH_PD_CFG`: SO-101 with a stiffer PD controller for task-space (IK) tracking.

The SO-101 is a low-cost 5-DOF arm (``shoulder_pan``, ``shoulder_lift``, ``elbow_flex``,
``wrist_flex``, ``wrist_roll``) plus a single revolute ``gripper`` jaw. Because the arm has
only 5 actuated DOF, it cannot achieve an arbitrary 6-DOF end-effector pose; task-space
controllers should command the full pose but soft-weight the orientation rows so position is
tracked exactly and orientation is best-effort (see the cube-stack IK-Abs task).

Reference: https://github.com/TheRobotStudio/SO-ARM100
Actuator gains follow the values tuned for simulation in LeIsaac.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the SO-101 follower arm with implicit actuators."""


SO101_HIGH_PD_CFG = SO101_CFG.copy()
SO101_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
SO101_HIGH_PD_CFG.actuators["arm"].stiffness = 400.0
SO101_HIGH_PD_CFG.actuators["arm"].damping = 80.0
"""Configuration of the SO-101 follower arm with stiffer PD control.

This configuration is useful for task-space control using differential IK, where the
default low-stiffness gains track end-effector targets poorly.
"""

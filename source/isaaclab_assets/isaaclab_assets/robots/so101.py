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
The default actuator parameters are authored in the USD. Multi-backend tasks select the ``physx``
physics variant for PhysX and the SysID ``physics`` variant for Newton MJWarp. The high-PD
configuration retains the gains previously tuned for IK tracking.
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
        usd_path=(f"{ISAAC_NUCLEUS_DIR}/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/so101_new_calib.usda"),
        variants={"Robot": "robot", "Sensor": "sensors", "Physics": "physics"},
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": -0.1221070742,
            "shoulder_lift": -0.9066845838,
            "elbow_flex": 0.1900876486,
            "wrist_flex": 1.4797928525,
            "wrist_roll": -0.8044013083,
            "gripper": 0.2555162024919699,
        },
    ),
    actuators={
        "usd": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },
    soft_joint_pos_limit_factor=0.98,
)
"""Configuration of the SO-101 follower arm using the USD-authored actuator parameters.

This standalone asset configuration selects the SysID ``physics`` USD variant for the default
Newton MJWarp backend. Preset-aware multi-backend tasks select ``physx`` when using PhysX.
"""


SO101_HIGH_PD_CFG = SO101_CFG.copy()
SO101_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
SO101_HIGH_PD_CFG.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        joint_effort_limit=10.0,
        joint_velocity_limit=10.0,
        stiffness=400.0,
        damping=80.0,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["gripper"],
        joint_effort_limit=10.0,
        joint_velocity_limit=10.0,
        stiffness=17.8,
        damping=0.60,
    ),
}
SO101_HIGH_PD_CFG.soft_joint_pos_limit_factor = 1.0
"""Configuration of the SO-101 follower arm with stiffer PD control.

This configuration is useful for task-space control using differential IK, where the
USD-authored SysID gains may track end-effector targets less tightly. Its actuator gains,
limits, gravity setting, and soft joint position limit factor match the previous high-PD
configuration.
"""

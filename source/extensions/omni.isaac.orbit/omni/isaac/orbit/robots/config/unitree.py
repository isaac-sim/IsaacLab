# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with simple PD controller for the legs

Reference: https://github.com/unitreerobotics/unitree_ros
"""


from omni.isaac.orbit.actuators.group import ActuatorControlCfg, ActuatorGroupCfg
from omni.isaac.orbit.actuators.model.actuator_cfg import DCMotorCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

from ..legged_robot import LeggedRobotCfg

##
# Configuration
##

_UNITREE_A1_INSTANCEABLE_USD = f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/Unitree/A1/a1_instanceable.usd"


UNITREE_A1_CFG = LeggedRobotCfg(
    meta_info=LeggedRobotCfg.MetaInfoCfg(usd_path=_UNITREE_A1_INSTANCEABLE_USD, soft_dof_pos_limit_factor=0.9),
    feet_info={
        "FR_foot": LeggedRobotCfg.FootFrameCfg(body_name="FR_calf", pos_offset=(0.0, 0.0, -0.2)),
        "FL_foot": LeggedRobotCfg.FootFrameCfg(body_name="FL_calf", pos_offset=(0.0, 0.0, -0.2)),
        "RR_foot": LeggedRobotCfg.FootFrameCfg(body_name="RR_calf", pos_offset=(0.0, 0.0, -0.2)),
        "RL_foot": LeggedRobotCfg.FootFrameCfg(body_name="RL_calf", pos_offset=(0.0, 0.0, -0.2)),
    },
    init_state=LeggedRobotCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        dof_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.8,
        },
        dof_vel={".*": 0.0},
    ),
    rigid_props=LeggedRobotCfg.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    collision_props=LeggedRobotCfg.CollisionPropertiesCfg(
        contact_offset=0.02,
        rest_offset=0.0,
    ),
    articulation_props=LeggedRobotCfg.ArticulationRootPropertiesCfg(
        enable_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
    ),
    actuator_groups={
        "base_legs": ActuatorGroupCfg(
            dof_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            model_cfg=DCMotorCfg(
                motor_torque_limit=33.5, gear_ratio=1.0, peak_motor_torque=33.5, motor_velocity_limit=21.0
            ),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 25.0},
                damping={".*": 0.5},
                dof_pos_offset={
                    ".*L_hip_joint": 0.1,
                    ".*R_hip_joint": -0.1,
                    "F[L,R]_thigh_joint": 0.8,
                    "R[L,R]_thigh_joint": 1.0,
                    ".*_calf_joint": -1.8,
                },
            ),
        )
    },
)
"""Configuration of Unitree A1 using DC motor.

Note: Specifications taken from: https://www.trossenrobotics.com/a1-quadruped#specifications
"""

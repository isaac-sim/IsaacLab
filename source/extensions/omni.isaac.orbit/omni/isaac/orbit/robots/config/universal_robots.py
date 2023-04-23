# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""


from omni.isaac.orbit.actuators.group import ActuatorGroupCfg
from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorControlCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

from ..single_arm import SingleArmManipulatorCfg

_UR10_ARM_INSTANCEBALE_USD = f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd"


UR10_CFG = SingleArmManipulatorCfg(
    meta_info=SingleArmManipulatorCfg.MetaInfoCfg(
        usd_path=_UR10_ARM_INSTANCEBALE_USD,
        arm_num_dof=6,
        tool_num_dof=0,
        tool_sites_names=None,
    ),
    init_state=SingleArmManipulatorCfg.InitialStateCfg(
        dof_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
        dof_vel={".*": 0.0},
    ),
    ee_info=SingleArmManipulatorCfg.EndEffectorFrameCfg(body_name="ee_link"),
    actuator_groups={
        "arm": ActuatorGroupCfg(
            dof_names=[".*"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=87.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": None},
                damping={".*": None},
            ),
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""

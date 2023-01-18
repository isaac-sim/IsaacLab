# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.actuators.group import ActuatorControlCfg, GripperActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg

"""
Actuator Groups.
"""

KINOVA_S300_MIMIC_GROUP_CFG = GripperActuatorGroupCfg(
    dof_names=[".*_joint_finger_[1-3]", ".*_joint_finger_tip_[1-3]"],
    model_cfg=ImplicitActuatorCfg(velocity_limit=1.0, torque_limit=2.0),
    control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 1e5}, damping={".*": 1e3}),
    mimic_multiplier={".*_joint_finger_[1-3]": 1.0, ".*_joint_finger_tip_[1-3]": 0.0},
    speed=0.1,
    open_dof_pos=0.65,
    close_dof_pos=0.0,
)
"""Configuration for Kinova S300 hand with implicit actuator model."""

KINOVA_S200_MIMIC_GROUP_CFG = GripperActuatorGroupCfg(
    dof_names=[".*_joint_finger_[1-2]", ".*_joint_finger_tip_[1-2]"],
    model_cfg=ImplicitActuatorCfg(velocity_limit=1.0, torque_limit=2.0),
    control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 1e5}, damping={".*": 1e3}),
    mimic_multiplier={".*_joint_finger_[1-2]": 1.0, ".*_joint_finger_tip_[1-2]": 0.0},
    speed=0.1,
    open_dof_pos=0.65,
    close_dof_pos=0.0,
)
"""Configuration for Kinova S200 hand with implicit actuator model."""

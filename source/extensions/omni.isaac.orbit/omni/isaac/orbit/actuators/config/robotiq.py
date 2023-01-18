# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.actuators.group import ActuatorControlCfg, GripperActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg

"""
Actuator Groups.
"""

ROBOTIQ_2F140_MIMIC_GROUP_CFG = GripperActuatorGroupCfg(
    dof_names=["finger_joint", ".*_inner_knuckle_joint", ".*_inner_finger_joint", ".*right_outer_knuckle_joint"],
    model_cfg=ImplicitActuatorCfg(velocity_limit=2.0, torque_limit=1000.0),
    control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 1e5}, damping={".*": 1e3}),
    mimic_multiplier={
        "finger_joint": 1.0,  # mimicked joint
        ".*_inner_knuckle_joint": -1.0,
        ".*_inner_finger_joint": 1.0,
        ".*right_outer_knuckle_joint": -1.0,
    },
    speed=0.01,
    open_dof_pos=0.7,
    close_dof_pos=0.0,
)
"""Configuration for Robotiq 2F-140 gripper with implicit actuator model."""

ROBOTIQ_2F85_MIMIC_GROUP_CFG = GripperActuatorGroupCfg(
    dof_names=["finger_joint", ".*_inner_knuckle_joint", ".*_inner_finger_joint", ".*right_outer_knuckle_joint"],
    model_cfg=ImplicitActuatorCfg(velocity_limit=2.0, torque_limit=1000.0),
    control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 1e5}, damping={".*": 1e3}),
    mimic_multiplier={
        "finger_joint": 1.0,  # mimicked joint
        ".*_inner_knuckle_joint": 1.0,
        ".*_inner_finger_joint": -1.0,
        ".*right_outer_knuckle_joint": 1.0,
    },
    speed=0.01,
    open_dof_pos=0.8,
    close_dof_pos=0.0,
)
"""Configuration for Robotiq 2F-85 gripper with implicit actuator model."""

ROBOTIQ_C2_MIMIC_GROUP_CFG = GripperActuatorGroupCfg(
    dof_names=[".*_left_knuckle_joint", ".*_right_knuckle_joint", ".*_inner_knuckle_joint", ".*_finger_tip_joint"],
    model_cfg=ImplicitActuatorCfg(velocity_limit=2.0, torque_limit=1000.0),
    control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 1e5}, damping={".*": 1e3}),
    mimic_multiplier={
        ".*_left_knuckle_joint": 1.0,  # mimicked joint
        ".*_right_knuckle_joint": 1.0,
        ".*_inner_knuckle_joint": 1.0,
        ".*_finger_tip_joint": -1.0,
    },
    speed=0.01,
    open_dof_pos=0.85,
    close_dof_pos=0.0,
)
"""Configuration for Robotiq C2 gripper with implicit actuator model."""

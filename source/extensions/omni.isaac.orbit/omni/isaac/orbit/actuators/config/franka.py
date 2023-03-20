# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.actuators.group import ActuatorControlCfg, GripperActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg

"""
Actuator Groups.
"""

PANDA_HAND_MIMIC_GROUP_CFG = GripperActuatorGroupCfg(
    dof_names=["panda_finger_joint[1-2]"],
    model_cfg=ImplicitActuatorCfg(velocity_limit=0.2, torque_limit=200),
    control_cfg=ActuatorControlCfg(command_types=["p_abs"], stiffness={".*": 1e5}, damping={".*": 1e3}),
    mimic_multiplier={"panda_finger_joint.*": 1.0},
    speed=0.1,
    open_dof_pos=0.04,
    close_dof_pos=0.0,
)
"""Configuration for Franka Panda hand with implicit actuator model."""

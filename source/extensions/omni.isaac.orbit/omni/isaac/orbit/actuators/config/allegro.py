# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.actuators.group import ActuatorControlCfg, GripperActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg

# TODO: Fix the configuration.
ALLEGRO_HAND_MIMIC_GROUP_CFG = GripperActuatorGroupCfg(
    dof_names=[".*finger_joint", ".*_inner_knuckle_joint", ".*_inner_finger_joint", ".*right_outer_knuckle_joint"],
    model_cfg=ImplicitActuatorCfg(velocity_limit=2.0, torque_limit=1000.0),
    control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 1e5}, damping={".*": 1e3}),
    mimic_multiplier={
        ".*finger_joint": 1.0,
        ".*_inner_knuckle_joint": -1.0,
        ".*_inner_finger_joint": 1.0,
        ".*right_outer_knuckle_joint": -1.0,
    },
    speed=0.1,
    open_dof_pos=0.87,
    close_dof_pos=0.0,
)
"""Configuration for Allegro hand with implicit actuator model."""

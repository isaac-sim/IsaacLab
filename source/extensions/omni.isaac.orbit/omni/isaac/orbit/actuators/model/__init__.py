# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains the configuration classes and explicit models for actuators."""

from .actuator_cfg import (
    ActuatorNetLSTMCfg,
    ActuatorNetMLPCfg,
    DCMotorCfg,
    IdealActuatorCfg,
    ImplicitActuatorCfg,
    VariableGearRatioDCMotorCfg,
)
from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from .actuator_physics import DCMotor, IdealActuator, VariableGearRatioDCMotor

__all__ = [
    # implicit
    "ImplicitActuatorCfg",
    # ideal actuator
    "IdealActuatorCfg",
    "IdealActuator",
    # dc motor
    "DCMotorCfg",
    "DCMotor",
    # variable gear
    "VariableGearRatioDCMotorCfg",
    "VariableGearRatioDCMotor",
    # mlp
    "ActuatorNetMLPCfg",
    "ActuatorNetMLP",
    # lstm
    "ActuatorNetLSTMCfg",
    "ActuatorNetLSTM",
]

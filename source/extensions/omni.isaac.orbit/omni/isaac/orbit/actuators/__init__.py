# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subpackage for handling actuator models."""

from __future__ import annotations

from .actuator_base import ActuatorBase
from .actuator_cfg import (
    ActuatorBaseCfg,
    ActuatorNetLSTMCfg,
    ActuatorNetMLPCfg,
    DCMotorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
)
from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from .actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator

__all__ = [
    # base actuator
    "ActuatorBase",
    "ActuatorBaseCfg",
    # implicit actuator
    "ImplicitActuatorCfg",
    "ImplicitActuator",
    # ideal pd actuator
    "IdealPDActuatorCfg",
    "IdealPDActuator",
    # dc motor
    "DCMotorCfg",
    "DCMotor",
    # actuator net -- lstm
    "ActuatorNetLSTMCfg",
    "ActuatorNetLSTM",
    # actuator net -- mlp
    "ActuatorNetMLPCfg",
    "ActuatorNetMLP",
]

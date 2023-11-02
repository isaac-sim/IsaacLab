# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Module containing different actuator groups.

- **default**: Direct control over the DOFs handled by the actuator group.
- **mimic**: Mimics given commands into each DOFs handled by actuator group.
- **non-holonomic**: Adds a 2D kinematics skid-steering constraint for the actuator group.
"""

from __future__ import annotations

from .actuator_control_cfg import ActuatorControlCfg
from .actuator_group import ActuatorGroup
from .actuator_group_cfg import ActuatorGroupCfg, GripperActuatorGroupCfg, NonHolonomicKinematicsGroupCfg

__all__ = [
    # control
    "ActuatorControlCfg",
    # default
    "ActuatorGroupCfg",
    "ActuatorGroup",
]

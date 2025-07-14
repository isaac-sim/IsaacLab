# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different actuator models.

Actuator models are used to model the behavior of the actuators in an articulation. These
are usually meant to be used in simulation to model different actuator dynamics and delays.

There are two main categories of actuator models that are supported:

- **Implicit**: Motor model with ideal PD from the physics engine. This is similar to having a continuous time
  PD controller. The motor model is implicit in the sense that the motor model is not explicitly defined by the user.
- **Explicit**: Motor models based on physical drive models.

  - **Physics-based**: Derives the motor models based on first-principles.
  - **Neural Network-based**: Learned motor models from actuator data.

Every actuator model inherits from the :class:`isaaclab.actuators.ActuatorBase` class,
which defines the common interface for all actuator models. The actuator models are handled
and called by the :class:`isaaclab.assets.Articulation` class.
"""

from .actuator_base import ActuatorBase
from .actuator_cfg import (
    ActuatorBaseCfg,
    ActuatorNetLSTMCfg,
    ActuatorNetMLPCfg,
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
    RemotizedPDActuatorCfg,
)
from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from .actuator_pd import DCMotor, DelayedPDActuator, IdealPDActuator, ImplicitActuator, RemotizedPDActuator

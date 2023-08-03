# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration instances of actuation models for ANYmal robot.

The following actuator models are available:

* ANYdrive 3.x with DC actuator model.
* ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model.

"""

from omni.isaac.orbit.actuators import ActuatorNetLSTMCfg, DCMotorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

"""
Actuator Models.
"""


@configclass
class Anydrive3SimpleCfg(DCMotorCfg):
    """Simple ANYdrive 3.0 model with only DC Motor limits."""

    saturation_effort: float = 120.0
    effort_limit: float = 80.0
    velocity_limit: float = 7.5


@configclass
class Anydrive3LSTMCfg(ActuatorNetLSTMCfg):
    """ANYdrive 3 model represented by an LSTM network trained from real data."""

    network_file: str = f"{ISAAC_ORBIT_NUCLEUS_DIR}/ActuatorNets/anydrive_3_lstm_jit.pt"
    saturation_effort: float = 120.0
    effort_limit: float = 80.0
    velocity_limit: float = 7.5

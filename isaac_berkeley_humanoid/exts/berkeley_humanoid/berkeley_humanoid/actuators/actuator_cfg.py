# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.actuators import DCMotorCfg

from .actuator_pd import IdentifiedActuator


@configclass
class IdentifiedActuatorCfg(DCMotorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = IdentifiedActuator

    friction_static: float = MISSING
    """ (in N-m)."""
    activation_vel: float = MISSING
    """ (in Rad/s)."""
    friction_dynamic: float = MISSING
    """ (in N-m-s/Rad)."""

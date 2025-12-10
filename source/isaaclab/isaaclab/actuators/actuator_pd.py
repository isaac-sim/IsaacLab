# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .actuator_base import ActuatorBase

if TYPE_CHECKING:
    from isaaclab_newton.actuators.actuator_pd import DCMotor as NewtonDCMotor
    from isaaclab_newton.actuators.actuator_pd import IdealPDActuator as NewtonIdealPDActuator
    from isaaclab_newton.actuators.actuator_pd import ImplicitActuator as NewtonImplicitActuator

    # from isaaclab_newton.actuators.actuator_pd import DelayedPDActuator as NewtonDelayedPDActuator
    # from isaaclab_newton.actuators.actuator_pd import RemotizedPDActuator as NewtonRemotizedPDActuator

"""
Implicit Actuator Models.
"""


class ImplicitActuator(FactoryBase):
    """Factory for creating implicit actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase | NewtonImplicitActuator:
        """Create a new instance of an implicit actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)


"""
Explicit Actuator Models.
"""


class IdealPDActuator(FactoryBase):
    """Factory for creating ideal PD actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase | NewtonIdealPDActuator:
        """Create a new instance of an ideal PD actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)


class DCMotor(FactoryBase):
    """Factory for creating DC motor actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase | NewtonDCMotor:
        """Create a new instance of a DC motor actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)


class DelayedPDActuator(FactoryBase):
    """Factory for creating delayed PD actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase:
        """Create a new instance of a delayed PD actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)


class RemotizedPDActuator(FactoryBase):
    """Factory for creating remotized PD actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase:
        """Create a new instance of a remotized PD actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)

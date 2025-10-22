# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.backend_utils import FactoryBase, Registerable

from .actuator_base import ActuatorBase

"""
Implicit Actuator Models.
"""


class ImplicitActuator(FactoryBase):
    """Factory for creating implicit actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase:
        """Create a new instance of an implicit actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)

class RegisterableImplicitActuator(Registerable):
    """A mixin to register an implicit actuator with the ImplicitActuator factory."""
    __factory_class__ = ImplicitActuator


"""
Explicit Actuator Models.
"""


class IdealPDActuator(FactoryBase):
    """Factory for creating ideal PD actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase:
        """Create a new instance of an ideal PD actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)


class RegisterableIdealPDActuator(Registerable):
    """A mixin to register an ideal PD actuator with the IdealPDActuator factory."""
    __factory_class__ = IdealPDActuator


class DCMotor(FactoryBase):
    """Factory for creating DC motor actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase:
        """Create a new instance of a DC motor actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)


class RegisterableDCMotor(Registerable):
    """A mixin to register a DC motor actuator with the DCMotor factory."""
    __factory_class__ = DCMotor


class DelayedPDActuator(FactoryBase):
    """Factory for creating delayed PD actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase:
        """Create a new instance of a delayed PD actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)


class RegisterableDelayedPDActuator(Registerable):
    """A mixin to register a delayed PD actuator with the DelayedPDActuator factory."""
    __factory_class__ = DelayedPDActuator


class RemotizedPDActuator(FactoryBase):
    """Factory for creating remotized PD actuator models."""

    def __new__(cls, *args, **kwargs) -> ActuatorBase:
        """Create a new instance of a remotized PD actuator model based on the backend."""
        return super().__new__(cls, *args, **kwargs)


class RegisterableRemotizedPDActuator(Registerable):
    """A mixin to register a remotized PD actuator with the RemotizedPDActuator factory."""
    __factory_class__ = RemotizedPDActuator
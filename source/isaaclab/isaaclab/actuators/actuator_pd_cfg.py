# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from . import actuator_pd
from .actuator_base_cfg import ActuatorBaseCfg

"""
Implicit Actuator Models.
"""


@configclass
class ImplicitActuatorCfg(ActuatorBaseCfg):
    """Configuration for an implicit actuator.

    Note:
        The PD control is handled implicitly by the simulation.
    """

    class_type: type = actuator_pd.ImplicitActuator


"""
Explicit Actuator Models.
"""


@configclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator."""

    class_type: type = actuator_pd.IdealPDActuator


@configclass
class DCMotorCfg(IdealPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = actuator_pd.DCMotor

    saturation_effort: float = MISSING
    """Peak motor force/torque of the electric DC motor (in N-m)."""


@configclass
class DelayedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = actuator_pd.DelayedPDActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


@configclass
class RemotizedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a remotized PD actuator.

    Note:
        The torque output limits for this actuator is derived from a linear interpolation of a lookup table
        in :attr:`joint_parameter_lookup`. This table describes the relationship between joint angles and
        the output torques.
    """

    class_type: type = actuator_pd.RemotizedPDActuator

    joint_parameter_lookup: list[list[float]] = MISSING
    """Joint parameter lookup table. Shape is (num_lookup_points, 3).

    This tensor describes the relationship between the joint angle (rad), the transmission ratio (in/out),
    and the output torque (N*m). The table is used to interpolate the output torque based on the joint angle.
    """

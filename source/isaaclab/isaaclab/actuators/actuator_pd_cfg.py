# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.utils import REQUIRED

from .actuator_base_cfg import ActuatorBaseCfg

if TYPE_CHECKING:
    from .actuator_pd import DCMotor, DelayedPDActuator, IdealPDActuator, ImplicitActuator, RemotizedPDActuator

"""
Implicit Actuator Models.
"""


@dataclass
class ImplicitActuatorCfg(ActuatorBaseCfg):
    """Configuration for an implicit actuator.

    Note:
        The PD control is handled implicitly by the simulation.
    """

    class_type: type["ImplicitActuator"] | str = "isaaclab.actuators.actuator_pd:ImplicitActuator"


"""
Explicit Actuator Models.
"""


@dataclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator."""

    class_type: type["IdealPDActuator"] | str = "isaaclab.actuators.actuator_pd:IdealPDActuator"


@dataclass
class DCMotorCfg(IdealPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type["DCMotor"] | str = "isaaclab.actuators.actuator_pd:DCMotor"

    saturation_effort: dict[str, float] | float = REQUIRED
    """Peak motor force/torque of the electric DC motor [N or N·m, depending on joint type].

    The motor's stall torque reflected at the joint, i.e. the torque produced at zero speed.
    Joints in the same group that sit behind different gear reductions need different values,
    so this accepts a joint-name-pattern dictionary as well as a scalar.
    """


@dataclass
class DelayedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type["DelayedPDActuator"] | str = "isaaclab.actuators.actuator_pd:DelayedPDActuator"

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


@dataclass
class RemotizedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a remotized PD actuator.

    Note:
        The torque output limits for this actuator is derived from a linear interpolation of a lookup table
        in :attr:`joint_parameter_lookup`. This table describes the relationship between joint angles and
        the output torques.
    """

    class_type: type["RemotizedPDActuator"] | str = "isaaclab.actuators.actuator_pd:RemotizedPDActuator"

    joint_parameter_lookup: list[list[float]] = REQUIRED
    """Joint parameter lookup table. Shape is (num_lookup_points, 3).

    This tensor describes the relationship between the joint angle (rad), the transmission ratio (in/out),
    and the output torque (N*m). The table is used to interpolate the output torque based on the joint angle.
    """

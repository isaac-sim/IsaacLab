# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import MISSING
from typing import TYPE_CHECKING

from ..utils import configclass
from ..utils.delay import DelayCfg
from .actuator_base_cfg import ActuatorBaseCfg

if TYPE_CHECKING:
    from .actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator, RemotizedPDActuator

"""
Implicit Actuator Models.
"""


@configclass
class ImplicitActuatorCfg(ActuatorBaseCfg):
    """Configuration for an implicit actuator.

    Note:
        The PD control is handled implicitly by the simulation.
    """

    class_type: type["ImplicitActuator"] | str = "{DIR}.actuator_pd:ImplicitActuator"


"""
Explicit Actuator Models.
"""


@configclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator."""

    class_type: type["IdealPDActuator"] | str = "{DIR}.actuator_pd:IdealPDActuator"


@configclass
class DCMotorCfg(IdealPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type["DCMotor"] | str = "{DIR}.actuator_pd:DCMotor"

    saturation_effort: dict[str, float] | float = MISSING
    """Peak motor force/torque of the electric DC motor [N or N·m, depending on joint type].

    The motor's stall torque reflected at the joint, i.e. the torque produced at zero speed.
    Joints in the same group that sit behind different gear reductions need different values,
    so this accepts a joint-name-pattern dictionary as well as a scalar.
    """


def DelayedPDActuatorCfg(*, min_delay: int = 0, max_delay: int = 0, **kwargs) -> DelayCfg:
    """Construct reset-sampled command delay around an ideal PD controller.

    Deprecated since 3.0; removed in 3.2. Use
    ``DelayCfg(term=IdealPDActuatorCfg(...), on="input", min_lag=..., max_lag=..., resample="reset")``.
    This constructor returns a :class:`~isaaclab.utils.DelayCfg`, not a separate actuator config type.
    Delay bounds count physics steps; remaining arguments configure the enclosed PD controller.
    """
    warnings.warn(
        "DelayedPDActuatorCfg is deprecated and will be removed in 3.2. Use "
        'DelayCfg(term=IdealPDActuatorCfg(...), on="input", min_lag=..., max_lag=..., resample="reset").',
        DeprecationWarning,
        stacklevel=2,
    )
    return DelayCfg(
        term=IdealPDActuatorCfg(**kwargs), on="input", min_lag=min_delay, max_lag=max_delay, resample="reset"
    )


@configclass
class RemotizedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a remotized PD actuator.

    Apply command latency with an enclosing :class:`~isaaclab.utils.DelayCfg`.

    Note:
        The torque output limits for this actuator is derived from a linear interpolation of a lookup table
        in :attr:`joint_parameter_lookup`. This table describes the relationship between joint angles and
        the output torques.
    """

    class_type: type["RemotizedPDActuator"] | str = "{DIR}.actuator_pd:RemotizedPDActuator"

    joint_parameter_lookup: list[list[float]] = MISSING
    """Joint parameter lookup table. Shape is (num_lookup_points, 3).

    This tensor describes the relationship between the joint angle (rad), the transmission ratio (in/out),
    and the output torque (N*m). The table is used to interpolate the output torque based on the joint angle.
    """

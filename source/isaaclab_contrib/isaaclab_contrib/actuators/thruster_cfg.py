# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import config_field

if TYPE_CHECKING:
    from .thruster import Thruster


@dataclass
class ThrusterCfg:
    """Configuration for thruster actuator groups.

    This config defines per-actuator-group parameters used by the low-level
    thruster/motor models (time-constants, thrust ranges, integration scheme,
    and initial state specifications). Fields left as ``MISSING`` are required
    and must be provided by the user configuration.
    """

    class_type: type["Thruster"] | str = config_field("{DIR}.thruster:Thruster")
    """Concrete Python class that consumes this config."""

    dt: float = config_field(MISSING)
    """Simulation/integration timestep used by the thruster update [s]."""

    thrust_range: tuple[float, float] = config_field(MISSING)
    """Per-motor thrust clamp range [N]: values are clipped to this interval."""

    max_thrust_rate: float = config_field(100000.0)
    """Per-motor thrust slew-rate limit applied inside the first-order model [N/s]."""

    thrust_const_range: tuple[float, float] = config_field(MISSING)
    """Range for thrust coefficient :math:`k_f` [N/(rps²)]."""

    tau_inc_range: tuple[float, float] = config_field(MISSING)
    """Range of time constants when commanded output is **increasing** (rise dynamics) [s]."""

    tau_dec_range: tuple[float, float] = config_field(MISSING)
    """Range of time constants when commanded output is **decreasing** (fall dynamics) [s]."""

    torque_to_thrust_ratio: float = config_field(MISSING)
    """Yaw-moment coefficient converting thrust to motor torque about +Z [N·m per N].
    Used as ``tau_z = torque_to_thrust_ratio * thrust_z * direction``.
    """

    use_discrete_approximation: bool = config_field(True)
    """
    Determines how the actuator/motor mixing factor is computed. Defaults to True.

    If True, uses the discrete-time factor ``1 / (dt + tau)``, accounting for the control loop timestep.
    If False, uses the continuous-time factor ``1 / tau``.
    """

    integration_scheme: Literal["rk4", "euler"] = config_field("rk4")
    """Numerical integrator for the first-order model. Defaults to 'rk4'."""

    thruster_names_expr: list[str] = config_field(MISSING)
    """Articulation's joint names that are part of the group."""

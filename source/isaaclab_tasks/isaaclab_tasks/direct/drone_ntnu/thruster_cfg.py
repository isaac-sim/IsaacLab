# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from typing import Literal

from isaaclab.utils import configclass

from .thruster import Thruster


@configclass
class ThrusterCfg:

    class_type: type[Thruster] = Thruster
    """Concrete Python class that consumes this config."""

    dt: float = 0.01
    """Simulation/integration timestep used by the thruster update [s]."""

    num_motors: int = 4
    """Number of motors/propulsors on the vehicle."""

    max_thrust: int = 2
    """Per-motor maximum thrust saturation [N]. Values above this are clipped."""

    max_thrust_rate: float = 100000.0
    """Per-motor thrust slew-rate limit applied inside the first-order model [N/s]."""

    min_thrust: int = 0
    """Per-motor minimum thrust [N]. Values below this are clipped."""

    thrust_const_max: float = 1.826312e-05
    """Upper bound for the thrust coefficient :math:`k_f` when ``use_rps=True`` [N/(rps²)]."""

    thrust_const_min: float = 9.26312e-06
    """Lower bound for the thrust coefficient :math:`k_f` when ``use_rps=True`` [N/(rps²)]."""

    tau_dec_max: float = 0.04
    """Maximum time constant used when the commanded output is **decreasing** (fall dynamics) [s]."""

    tau_dec_min: float = 0.04
    """Minimum time constant used when the commanded output is **decreasing** (fall dynamics) [s]."""

    tau_inc_max: float = 0.04
    """Maximum time constant used when the commanded output is **increasing** (rise dynamics) [s]."""

    tau_inc_min: float = 0.04
    """Minimum time constant used when the commanded output is **increasing** (rise dynamics) [s]."""

    thrust_to_torque_ratio: float = 0.01
    """Yaw-moment coefficient converting thrust to motor torque about +Z [N·m per N].
    Used as ``tau_z = thrust_to_torque_ratio * thrust_z * direction``.
    """

    use_discrete_approximation: bool = True
    """If ``True``, use discrete mixing factor ``1/(dt + tau)``; if ``False``, use continuous ``1/tau``."""

    use_rps: bool = True
    """If ``True``, integrate in rotor-speed domain (ω) and compute thrust via ``F = k_f * ω**2``.
    If ``False``, integrate thrust directly.
    """

    integration_scheme: Literal["rk4", "euler"] = "rk4"
    """Numerical integrator for the first-order model. Choose ``"euler"`` or ``"rk4"``."""
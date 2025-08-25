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

    dt: float = 0.01

    num_motors: int = 4
    """Number of motors in the drone"""

    max_thrust: int = 2

    max_thrust_rate: float = 100000.0

    min_thrust: int = 0

    motor_thrust_constant_max: float = 1.826312e-05

    motor_thrust_constant_min: float = 9.26312e-06

    motor_time_constant_decreasing_max: float = 0.04

    motor_time_constant_decreasing_min: float = 0.04

    motor_time_constant_increasing_max: float = 0.04

    motor_time_constant_increasing_min: float = 0.04

    thrust_to_torque_ratio: float = 0.01

    use_discrete_approximation: bool = True

    use_rps: bool = True

    integration_scheme: Literal["rk4", "euler"] = "rk4"

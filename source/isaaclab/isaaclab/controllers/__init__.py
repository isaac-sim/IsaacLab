# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different controllers and motion-generators.

Controllers or motion generators are responsible for closed-loop tracking of a given command. The
controller can be a simple PID controller or a more complex controller such as impedance control
or inverse kinematics control. The controller is responsible for generating the desired joint-level
commands to be sent to the robot.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

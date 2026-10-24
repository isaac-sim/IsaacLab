# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad device for SE(2) and SE(3) control.

.. deprecated::
    Use :func:`isaaclab_teleop.se2_gamepad_teleop_cfg` / :func:`isaaclab_teleop.se3_gamepad_teleop_cfg`, which run
    the device in process through Isaac Capture. Imports from this package keep working;
    the device constructors emit a :class:`DeprecationWarning`.
"""

from ...utils.module import lazy_export

lazy_export()

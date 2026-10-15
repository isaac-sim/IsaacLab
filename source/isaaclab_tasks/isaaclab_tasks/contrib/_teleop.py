# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared availability check for the optional Isaac Lab Teleop package."""

try:
    import isaaclab_teleop  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab_teleop":
        raise
    ISAACLAB_TELEOP_AVAILABLE = False
else:
    ISAACLAB_TELEOP_AVAILABLE = True

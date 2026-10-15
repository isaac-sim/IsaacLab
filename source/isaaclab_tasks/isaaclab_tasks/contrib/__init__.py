# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contributed task families.

These may depend on :mod:`isaaclab_tasks.core` tasks, but ``core`` tasks never depend
on ``contrib`` tasks.
"""

from importlib.util import find_spec

ISAACLAB_TELEOP_AVAILABLE = find_spec("isaaclab_teleop") is not None
"""Whether the optional Isaac Lab Teleop package is installed."""

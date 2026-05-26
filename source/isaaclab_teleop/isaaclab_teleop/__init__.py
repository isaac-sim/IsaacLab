# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package providing IsaacTeleop-based teleoperation for Isaac Lab."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("isaaclab_teleop")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from isaaclab.utils.module import lazy_export

lazy_export()

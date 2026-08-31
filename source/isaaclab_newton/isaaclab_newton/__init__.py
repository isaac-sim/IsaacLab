# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the Newton simulation interfaces for IsaacLab core package."""

import importlib.metadata

import newton

# Newton reads this while a ModelBuilder is populated and finalized, so it has to
# be set before any builder exists. Isaac Lab indexes joint position targets per DOF.
# Deprecated since Newton 1.5; pinned until Isaac Lab moves to the coordinate layout.
newton.use_coord_layout_targets = False

try:
    __version__ = importlib.metadata.version("isaaclab_newton")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

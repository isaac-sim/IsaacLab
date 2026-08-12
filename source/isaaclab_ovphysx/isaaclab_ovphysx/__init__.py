# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the ovphysx/TensorBindingsAPI simulation interfaces for IsaacLab."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("isaaclab_ovphysx")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

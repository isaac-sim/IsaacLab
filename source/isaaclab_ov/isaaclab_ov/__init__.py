# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing Omniverse renderers for IsaacLab (OVRTX, ovphysx, etc.)."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("isaaclab_ov")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

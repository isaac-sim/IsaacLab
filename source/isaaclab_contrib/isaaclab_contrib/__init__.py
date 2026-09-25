# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package for externally contributed components for Isaac Lab.

This package provides externally contributed components for Isaac Lab, such as multirotors.
These components are not part of the core Isaac Lab framework yet, but are planned to be added
in the future. They are contributed by the community to extend the capabilities of Isaac Lab.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("isaaclab_contrib")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

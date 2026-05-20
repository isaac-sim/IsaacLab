# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for externally contributed assets.

This package contains contributed code that depends on Isaac Lab's public API but is not required for core functionality. This includes implementations of Newton solvers for deformables.
"""

from isaaclab.utils.module import lazy_export

lazy_export()

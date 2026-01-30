# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for externally contributed assets.

This package provides specialized asset classes for simulating externally contributed
robots in Isaac Lab, such as multirotors. These assets are not part of the core
Isaac Lab framework yet, but are planned to be added in the future. They are
contributed by the community to extend the capabilities of Isaac Lab.
"""

from .multirotor import Multirotor, MultirotorCfg, MultirotorData

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Backward compatibility shim for retargeters.

.. warning::
    This module is deprecated. Please use :mod:`isaaclab.devices.retargeters` instead.
"""

import warnings

from isaaclab.devices.retargeters import *  # noqa: F401, F403

warnings.warn(
    "isaaclab.devices.openxr.retargeters has moved to isaaclab.devices.retargeters. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

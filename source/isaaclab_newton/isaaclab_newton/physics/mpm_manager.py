# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated alias of :mod:`isaaclab_newton.physics.mpm_solver`.

.. deprecated::
    Import from :mod:`isaaclab_newton.physics.mpm_solver` or the package instead.
"""

import warnings

from .mpm_solver import *  # noqa: F401, F403

warnings.warn(
    "isaaclab_newton.physics.mpm_manager is deprecated; import from isaaclab_newton.physics.mpm_solver instead.",
    DeprecationWarning,
    stacklevel=2,
)

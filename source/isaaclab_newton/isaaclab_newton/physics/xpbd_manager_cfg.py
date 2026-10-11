# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated alias of :mod:`isaaclab_newton.physics.xpbd_solver_cfg`.

.. deprecated::
    Import from :mod:`isaaclab_newton.physics.xpbd_solver_cfg` or the package instead.
"""

import warnings

from .xpbd_solver_cfg import *  # noqa: F401, F403

warnings.warn(
    "isaaclab_newton.physics.xpbd_manager_cfg is deprecated; "
    "import from isaaclab_newton.physics.xpbd_solver_cfg instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated import path for :mod:`isaaclab_newton.physics.coupling`."""

import warnings
from isaaclab_newton.physics.coupling import (
    CouplerAdmmCfg,
    CouplerCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
    CouplerSolver,
)

__all__ = [
    "CouplerAdmmCfg",
    "CouplerCfg",
    "CouplerEntryCfg",
    "CouplerProxyCfg",
    "CouplerProxyMappingCfg",
    "CouplerSolver",
]

warnings.warn(
    "isaaclab_contrib.coupling is deprecated. Use isaaclab_newton.physics.coupling instead.",
    DeprecationWarning,
    stacklevel=2,
)

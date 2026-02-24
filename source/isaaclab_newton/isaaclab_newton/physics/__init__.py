# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation backends for simulation interfaces."""

import lazy_loader as lazy

from .newton_manager_cfg import (
    FeatherstoneSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonSolverCfg,
    XPBDSolverCfg,
)

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "newton_manager": ["NewtonManager"],
    },
)
__all__ += ["FeatherstoneSolverCfg", "MJWarpSolverCfg", "NewtonCfg", "NewtonSolverCfg", "XPBDSolverCfg"]

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation backends for simulation interfaces."""

import lazy_loader as lazy

from .physx_manager_cfg import PhysxCfg

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "physx_manager": ["PhysxManager", "IsaacEvents"],
    },
)
__all__ += ["PhysxCfg"]

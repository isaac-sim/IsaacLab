# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

<<<<<<< mtrepte/add_rendering_quality_cfg
"""Newton-backed renderer configurations and implementations."""
=======
"""Sub-module for Newton renderer backends (Newton Warp)."""
>>>>>>> develop

from .newton_warp_renderer import NewtonWarpRenderer
from .newton_warp_renderer_cfg import NewtonWarpRendererCfg

<<<<<<< mtrepte/add_rendering_quality_cfg
__all__ = ["NewtonWarpRenderer", "NewtonWarpRendererCfg"]
=======
Renderer = NewtonWarpRenderer

__all__ = [
    "NewtonWarpRenderer",
    "NewtonWarpRendererCfg",
    "Renderer",
]
>>>>>>> develop

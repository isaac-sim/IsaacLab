# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

<<<<<<< mtrepte/add_rendering_quality_cfg
"""PhysX-backed renderer configurations and implementations."""

from .rtx_renderer_cfg import RTXRendererCfg

__all__ = ["RTXRendererCfg"]
=======
"""Sub-module for PhysX renderer backends (Isaac RTX / Omniverse Replicator)."""

from .isaac_rtx_renderer import IsaacRtxRenderer
from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

Renderer = IsaacRtxRenderer

__all__ = [
    "IsaacRtxRenderer",
    "IsaacRtxRendererCfg",
    "Renderer",
]
>>>>>>> develop

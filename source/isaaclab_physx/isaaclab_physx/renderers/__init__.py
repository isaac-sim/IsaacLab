# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for PhysX renderer backends (Isaac RTX / Omniverse Replicator)."""

from .isaac_rtx_renderer import IsaacRtxRenderer
from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg
from .isaac_rtx_renderer_utils import SIMPLE_SHADING_MODES

Renderer = IsaacRtxRenderer

__all__ = [
    "IsaacRtxRenderer",
    "IsaacRtxRendererCfg",
    "Renderer",
    "SIMPLE_SHADING_MODES",
]

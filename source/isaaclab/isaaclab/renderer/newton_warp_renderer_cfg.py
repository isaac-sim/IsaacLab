# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton Warp Renderer."""

from isaaclab.utils import configclass

from .renderer_cfg import RendererCfg


@configclass
class NewtonWarpRendererCfg(RendererCfg):
    """Configuration for the Newton Warp renderer.

    Use with ``TiledCameraCfg(renderer_type="warp_renderer", ...)`` for Warp-based ray tracing
    alongside PhysX simulation. Requires the ``newton`` package.
    """

    renderer_type: str = "warp_renderer"
    """Type identifier for the Newton Warp renderer."""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton Warp Renderer."""

from isaaclab.utils import configclass

from .renderer_cfg import RendererCfg


@configclass
class NewtonWarpRendererCfg(RendererCfg):
    """Configuration for Newton Warp Renderer."""

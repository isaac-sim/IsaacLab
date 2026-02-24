# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton Warp Renderer."""

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass


@configclass
class NewtonWarpRendererCfg(RendererCfg):
    """Configuration for Newton Warp Renderer."""

    renderer_type: str = "newton_warp"
    """Type identifier for Newton Warp renderer."""

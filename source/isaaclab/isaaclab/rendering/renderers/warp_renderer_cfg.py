# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stub config for future Warp/Newton renderer integration."""

from isaaclab.utils import configclass

from .renderer_cfg import RendererCfg


@configclass
class WarpRendererCfg(RendererCfg):
    """Stub config for future Warp/Newton renderer integration.

    TODO: Implement renderer lifecycle, sensor/render-product routing, and
    backend-specific settings application.
    """

    renderer_type: str = "warp"

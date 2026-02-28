# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

<<<<<<< mtrepte/add_rendering_quality_cfg
"""Stub config for future Warp/Newton renderer integration."""

from isaaclab.renderers import RendererCfg
=======
"""Configuration for Newton Warp Renderer."""

from isaaclab.renderers.renderer_cfg import RendererCfg
>>>>>>> develop
from isaaclab.utils import configclass


@configclass
class NewtonWarpRendererCfg(RendererCfg):
<<<<<<< mtrepte/add_rendering_quality_cfg
    """Stub config for future Warp/Newton renderer integration.

    TODO: Implement renderer lifecycle, sensor/render-product routing, and
    backend-specific settings application.
    """

    renderer_type: str = "warp"
=======
    """Configuration for Newton Warp Renderer."""

    renderer_type: str = "newton_warp"
    """Type identifier for Newton Warp renderer."""
>>>>>>> develop

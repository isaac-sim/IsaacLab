# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stub renderer config classes.

These are intentionally lightweight placeholders so the rendering domain can
stabilize around explicit renderer config names before full implementation.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class RendererCfg:
    """Base configuration for all renderer backends.

    Note:
        This is an abstract base class and should not be instantiated directly.
        Use specific renderer configs like RTXRendererCfg or WarpRendererCfg.
    """

    renderer_type: str | None = None
    """Type identifier (e.g., 'rtx', 'warp'). Must be overridden by subclasses."""

    rendering_quality: str | None = None
    """Name of the rendering quality profile to use with this renderer."""


@configclass
class RTXRendererCfg(RendererCfg):
    """Stub config for future RTX renderer integration.

    TODO: Implement renderer lifecycle, sensor/render-product routing, and
    backend-specific settings application.
    """

    renderer_type: str = "rtx"
    rendering_quality: str | None = None


@configclass
class WarpRendererCfg(RendererCfg):
    """Stub config for future Warp/Newton renderer integration.

    TODO: Implement renderer lifecycle, sensor/render-product routing, and
    backend-specific settings application.
    """

    renderer_type: str = "warp"

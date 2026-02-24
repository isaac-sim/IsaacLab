# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stub renderer config classes.

These are intentionally lightweight placeholders so the rendering domain can
stabilize around explicit renderer config names before full implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from typing import Any


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

    def get_renderer_type(self) -> str | None:
        """Get the renderer type identifier."""
        return self.renderer_type

    def create_renderer(self) -> Any:
        """Create renderer instance using the renderer registry.

        TODO: Replace Any with concrete renderer protocol/base class once runtime
        renderer implementations are introduced.
        """
        from . import get_renderer_class

        if self.renderer_type is None:
            raise ValueError(
                "Cannot create renderer from base RendererCfg class. "
                "Use a specific renderer config: RTXRendererCfg or WarpRendererCfg."
            )

        renderer_class = get_renderer_class(self.renderer_type)
        if renderer_class is None:
            raise ValueError(
                f"Renderer type '{self.renderer_type}' is not registered. "
                "Valid types: 'rtx', 'warp'."
            )
        return renderer_class(self)


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

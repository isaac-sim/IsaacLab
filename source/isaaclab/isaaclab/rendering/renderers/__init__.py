# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer runtime backends namespace.

This package is intentionally created now as part of rendering-domain consolidation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .renderer_cfg import RendererCfg
from .rtx_renderer_cfg import RTXRendererCfg
from .warp_renderer_cfg import WarpRendererCfg

if TYPE_CHECKING:
    from .newton_warp_renderer import NewtonWarpRenderer
    from .renderer import Renderer

_RENDERER_REGISTRY: dict[str, Any] = {}

__all__ = [
    "Renderer",
    "NewtonWarpRenderer",
    "RendererCfg",
    "RTXRendererCfg",
    "WarpRendererCfg",
    "get_renderer_class",
]


def get_renderer_class(name: str) -> Any | None:
    """Get a renderer class by name (lazy-loaded).

    TODO: Wire real renderer runtime classes into this registry.
    """
    if name in _RENDERER_REGISTRY:
        return _RENDERER_REGISTRY[name]
    return None


def __getattr__(name: str) -> Any:
    """Lazy-load runtime renderer classes to avoid import cycles."""
    if name == "Renderer":
        from .renderer import Renderer

        return Renderer
    if name == "NewtonWarpRenderer":
        from .newton_warp_renderer import NewtonWarpRenderer

        return NewtonWarpRenderer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

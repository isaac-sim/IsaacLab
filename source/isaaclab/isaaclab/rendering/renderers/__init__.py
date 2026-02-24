# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer runtime backends namespace.

This package is intentionally created now as part of rendering-domain consolidation.
"""

from __future__ import annotations

from typing import Any

from .renderer_cfg import RTXRendererCfg, RendererCfg, WarpRendererCfg

_RENDERER_REGISTRY: dict[str, Any] = {}

__all__ = ["RendererCfg", "RTXRendererCfg", "WarpRendererCfg", "get_renderer_class"]


def get_renderer_class(name: str) -> Any | None:
    """Get a renderer class by name (lazy-loaded).

    TODO: Wire real renderer runtime classes into this registry.
    """
    if name in _RENDERER_REGISTRY:
        return _RENDERER_REGISTRY[name]
    return None

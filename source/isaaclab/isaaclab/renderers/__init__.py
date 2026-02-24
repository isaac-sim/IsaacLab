# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for renderer configurations and implementations.

This sub-package contains configuration classes and implementations for
different renderer backends that can be used with Isaac Lab.
"""

from __future__ import annotations

from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg
from .newton_warp_renderer_cfg import NewtonWarpRendererCfg

from .renderer import Renderer
from .renderer_cfg import RendererCfg

# Cache renderer class by config class (lazy-loaded)
_RENDERER_REGISTRY: dict[type[RendererCfg], type[Renderer]] = {}

__all__ = [
    "Renderer",
    "RendererCfg",
    "IsaacRtxRendererCfg",
    "NewtonWarpRendererCfg",
    "create_renderer",
]


def get_renderer_class(cfg: RendererCfg) -> type[Renderer]:
    """Get the renderer class for the given config instance (lazy-loaded).

    Dispatch is by config type (isinstance); no string lookup. Each config class
    maps to its renderer implementation.

    Raises:
        ValueError: If the config type is not registered.
        ImportError: If the renderer module fails to load.
    """
    cfg_type = type(cfg)
    if cfg_type in _RENDERER_REGISTRY:
        return _RENDERER_REGISTRY[cfg_type]
    if isinstance(cfg, IsaacRtxRendererCfg):
        from .isaac_rtx_renderer import IsaacRtxRenderer

        _RENDERER_REGISTRY[IsaacRtxRendererCfg] = IsaacRtxRenderer
        return IsaacRtxRenderer
    if isinstance(cfg, NewtonWarpRendererCfg):
        from .newton_warp_renderer import NewtonWarpRenderer

        _RENDERER_REGISTRY[NewtonWarpRendererCfg] = NewtonWarpRenderer
        return NewtonWarpRenderer
    raise ValueError(f"No renderer registered for config type '{cfg_type.__name__}'.")


def create_renderer(cfg: RendererCfg) -> Renderer:
    """Create a renderer instance from config.

    Args:
        cfg: Renderer configuration.

    Returns:
        The renderer instance.
    """
    renderer_cls = get_renderer_class(cfg)
    return renderer_cls(cfg)

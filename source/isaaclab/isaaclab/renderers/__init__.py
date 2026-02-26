# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for renderer configurations and implementations.

Renderer registry and Hydra:
- **renderer_cfg_from_type(renderer_type)** maps string → Renderer config instance
  ("newton_warp" → NewtonWarpRendererCfg(), "isaac_rtx"/None → IsaacRtxRendererCfg()).
- **get_renderer_class(name_or_cfg)** returns the renderer class for a config or name.
- TiledCamera uses **Renderer(cfg)** or resolves cfg from renderer_type in _initialize_impl().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from .base_renderer import BaseRenderer
from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg
from .renderer import Renderer
from .renderer_cfg import RendererCfg


def renderer_cfg_from_type(renderer_type: str | None) -> RendererCfg:
    """Map Hydra/CLI renderer_type string to a renderer config.

    Used so that ``env.scene.base_camera.renderer_type=newton_warp`` (or ``=isaac_rtx``)
    works: TiledCamera resolves renderer_cfg from this in _initialize_impl().

    Args:
        renderer_type: "newton_warp" → NewtonWarpRendererCfg();
            "isaac_rtx" or None → IsaacRtxRendererCfg().

    Returns:
        The corresponding config instance.
    """
    if renderer_type == "newton_warp":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg
        return NewtonWarpRendererCfg()
    return IsaacRtxRendererCfg()


def get_renderer_class(name_or_cfg: Union[str, RendererCfg]) -> type[BaseRenderer] | None:
    """Return renderer class for the given name or config. Prefer using Renderer(cfg) factory."""
    if isinstance(name_or_cfg, RendererCfg):
        try:
            from isaaclab_newton.renderers import NewtonWarpRendererCfg
            if isinstance(name_or_cfg, NewtonWarpRendererCfg):
                from isaaclab_newton.renderers import NewtonWarpRenderer
                return NewtonWarpRenderer
        except ImportError:
            pass
        if isinstance(name_or_cfg, IsaacRtxRendererCfg):
            from isaaclab_physx.renderers import IsaacRtxRenderer
            return IsaacRtxRenderer
        name_or_cfg = getattr(name_or_cfg, "renderer_type", None) or "isaac_rtx"
    name = name_or_cfg
    if name == "newton_warp":
        from isaaclab_newton.renderers import NewtonWarpRenderer
        return NewtonWarpRenderer
    if name == "isaac_rtx":
        from isaaclab_physx.renderers import IsaacRtxRenderer
        return IsaacRtxRenderer
    return None


__all__ = [
    "BaseRenderer",
    "Renderer",
    "RendererCfg",
    "IsaacRtxRendererCfg",
    "get_renderer_class",
    "renderer_cfg_from_type",
]

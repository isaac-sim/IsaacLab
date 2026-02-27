# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for creating renderer instances."""

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_renderer import BaseRenderer
from .renderer_cfg import RendererCfg

# This is mapping of where backends live in the isaaclab_<backend> package.
_RENDERER_TYPE_TO_BACKEND = {"isaac_rtx": "physx", "newton_warp": "newton"}


class Renderer(FactoryBase, BaseRenderer):
    """Factory for creating renderer instances."""

    @classmethod
    def _get_backend(cls, cfg: RendererCfg, *args, **kwargs) -> str:
        return _RENDERER_TYPE_TO_BACKEND.get(cfg.renderer_type, "physx")

    def __new__(cls, cfg: RendererCfg, *args, **kwargs) -> BaseRenderer:
        """Create a new instance of a renderer based on the backend."""
        # The `FactoryBase` __new__ method will handle the logic and return
        # an instance of the correct backend-specific renderer class,
        # which is guaranteed to be a subclass of `BaseRenderer` by convention.
        return super().__new__(cls, cfg, *args, **kwargs)


def renderer_cfg_from_type(renderer_type: str | None) -> RendererCfg:
    """Map renderer_type string to a renderer config instance.

    Used by TiledCamera._get_effective_renderer_cfg() (fallback if Hydra arg did not set renderer_cfg
    for that camera).

    Args:
        renderer_type: "newton_warp" → Newton backend config;
            "isaac_rtx" or None → PhysX (Isaac RTX) backend config.

    Returns:
        The corresponding config instance.
    """
    if renderer_type == "newton_warp":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg
        return NewtonWarpRendererCfg()
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    return IsaacRtxRendererCfg()

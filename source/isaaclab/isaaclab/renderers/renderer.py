# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for creating renderer instances."""

from __future__ import annotations

import importlib
from typing import Union

from isaaclab.utils.backend_utils import FactoryBase

from .base_renderer import BaseRenderer
from .renderer_cfg import RendererCfg

# This is mapping of where backends live in the isaaclab_<backend> package.
_RENDERER_TYPE_TO_BACKEND = {"isaac_rtx": "physx", "newton_warp": "newton"}


class Renderer(FactoryBase, BaseRenderer):
    """Factory for creating renderer instances."""

    @classmethod
    def _get_backend(cls, cfg: RendererCfg, *args, **kwargs) -> str:
        rt = getattr(cfg, "renderer_type", None)
        return _RENDERER_TYPE_TO_BACKEND.get(rt, "physx")

    def __new__(cls, cfg: RendererCfg, *args, **kwargs) -> BaseRenderer:
        """Create a new instance of a renderer based on the backend."""
        # The `FactoryBase` __new__ method will handle the logic and return
        # an instance of the correct backend-specific renderer class,
        # which is guaranteed to be a subclass of `BaseRenderer` by convention.
        return super().__new__(cls, cfg, *args, **kwargs)


def renderer_cfg_from_type(renderer_type: str | None) -> RendererCfg:
    """Map renderer_type string to a renderer config instance.

    Used by isaaclab_tasks.utils.hydra.instantiate_renderer_cfg_in_env() and by
    TiledCamera._get_effective_renderer_cfg() (fallback for non-Hydra paths).

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


def get_renderer_class(name_or_cfg: Union[str, RendererCfg]) -> type[BaseRenderer] | None:
    """Return the renderer implementation class for the given name or config.

    Prefer using Renderer(cfg) to create instances; this is for callers that
    need the class (e.g. for type checks or subclassing).

    Args:
        name_or_cfg: Renderer type string ("isaac_rtx", "newton_warp") or
            a RendererCfg instance (backend is taken from renderer_type).

    Returns:
        The backend renderer class, or None if the type is unknown.
    """
    if isinstance(name_or_cfg, RendererCfg):
        rt = getattr(name_or_cfg, "renderer_type", None) or "isaac_rtx"
    else:
        rt = name_or_cfg if name_or_cfg else "isaac_rtx"
    backend = _RENDERER_TYPE_TO_BACKEND.get(rt, "physx")
    mod = importlib.import_module(f"isaaclab_{backend}.renderers")
    return getattr(mod, "Renderer", None)

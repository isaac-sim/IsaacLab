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
_RENDERER_TYPE_TO_BACKEND = {"isaac_rtx": "physx", "newton_warp": "newton", "ov_rtx": "ovrtx"}


class Renderer(FactoryBase, BaseRenderer):
    """Factory for creating renderer instances."""

    _backend_class_names = {
        "physx": "IsaacRtxRenderer",
        "newton": "NewtonWarpRenderer",
        "ovrtx": "OVRTXRenderer",
    }

    @classmethod
    def _get_backend(cls, cfg: RendererCfg, *args, **kwargs) -> str:
        return _RENDERER_TYPE_TO_BACKEND.get(cfg.renderer_type, "physx")

    def __new__(cls, cfg: RendererCfg, *args, **kwargs) -> BaseRenderer:
        """Create a new instance of a renderer based on the backend."""
        # The `FactoryBase` __new__ method will handle the logic and return
        # an instance of the correct backend-specific renderer class.
        result = super().__new__(cls, cfg, *args, **kwargs)
        if not isinstance(result, BaseRenderer):
            name = type(result).__name__
            bases = type(result).__bases__
            raise TypeError(
                f"Backend renderer {name!r} must inherit from BaseRenderer, but it inherits from {bases!r}."
            )
        return result

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for creating renderer instances (develop/Main_Fork style).

TiledCamera uses Renderer(cfg) to get either IsaacRtxRenderer (physx) or NewtonWarpRenderer (newton).
Hydra/task sets cfg via renderer_cfg on the camera (e.g. from renderer_type string).
"""

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_renderer import BaseRenderer
from .renderer_cfg import RendererCfg

# Backend package names for dynamic loading (isaaclab_physx, isaaclab_newton).
_RENDERER_TYPE_TO_BACKEND = {
    "isaac_rtx": "physx",
    "newton_warp": "newton",
    "warp_renderer": "newton",
}


class Renderer(FactoryBase, BaseRenderer):
    """Factory for creating renderer instances. Use with TiledCamera: Renderer(self.cfg.renderer_cfg)."""

    @classmethod
    def _get_backend(cls, cfg: RendererCfg, *args, **kwargs) -> str:
        return _RENDERER_TYPE_TO_BACKEND.get(getattr(cfg, "renderer_type", None), "physx")

    def __new__(cls, cfg: RendererCfg, *args, **kwargs) -> BaseRenderer:
        """Create a new instance of a renderer based on the backend."""
        return super().__new__(cls, cfg, *args, **kwargs)

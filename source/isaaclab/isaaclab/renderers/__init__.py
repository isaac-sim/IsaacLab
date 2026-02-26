# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for renderer configurations and implementations.

Renderer registry and config resolution:
- **renderer_cfg_from_type(renderer_type)** maps string → Renderer config instance.
- **get_renderer_class(name_or_cfg)** returns the renderer class for a config or name.
- TiledCamera uses **Renderer(cfg)** or resolves cfg from renderer_type in _initialize_impl().
"""

from __future__ import annotations

from .base_renderer import BaseRenderer
from .renderer import Renderer, get_renderer_class, renderer_cfg_from_type
from .renderer_cfg import RendererCfg


__all__ = [
    "BaseRenderer",
    "Renderer",
    "RendererCfg",
    "get_renderer_class",
    "renderer_cfg_from_type",
]

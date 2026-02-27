# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for renderer configurations and implementations.

Renderer registry and config resolution:
- **renderer_cfg_from_type(renderer_type)** maps string → Renderer config instance (used by Hydra and TiledCamera fallback).
- When using Hydra (e.g. train.py), renderer_cfg is instantiated in isaaclab_tasks.utils.hydra before env creation.
- TiledCamera uses **Renderer(cfg)**; for non-Hydra paths it uses cfg.renderer_cfg or isaac_rtx fallback.
"""

from __future__ import annotations

from .base_renderer import BaseRenderer
from .renderer import Renderer, renderer_cfg_from_type
from .renderer_cfg import RendererCfg


__all__ = [
    "BaseRenderer",
    "Renderer",
    "RendererCfg",
    "renderer_cfg_from_type",
]

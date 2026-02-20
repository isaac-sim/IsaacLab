# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for renderer configurations and implementations.
This sub-package contains configuration classes and implementations for different
renderer backends that can be used with Isaac Lab. The renderers are used for
debug visualization and monitoring of the simulation, separate from rendering for sensors.
Supported visualizers:
- Newton Warp Renderer: Newton Warp-based renderer
- Omniverse RTX Renderer: High-fidelity Omniverse-based renderer.
- Kit App Renderer: Renderer that uses the Kit App to render the scene.
Visualizer Registry
-------------------
This module uses a registry pattern to decouple renderer instantiation from specific types.
Renderer implementations can register themselves using the `register_renderer` decorator,
and configs can create renderers via the `create_renderer()` factory method.
"""

from __future__ import annotations

# Import config classes (no circular dependency)
from .newton_warp_renderer_cfg import NewtonWarpRendererCfg
from .ovrtx_renderer_cfg import OVRTXRendererCfg

# Import base classes first
from .renderer import RendererBase
from .renderer_cfg import RendererCfg

# from .kit_app_renderer_cfg import KitAppRendererCfg

# Cache renderer class by config class (lazy-loaded)
_RENDERER_REGISTRY: dict[type[RendererCfg], type[RendererBase]] = {}

__all__ = [
    "RendererBase",
    "RendererCfg",
    "NewtonWarpRendererCfg",
    "OVRTXRendererCfg",
    "get_renderer_class",
]


def get_renderer_class(cfg: RendererCfg) -> type[RendererBase]:
    """Get the renderer class for the given config instance (lazy-loaded).

    Dispatch is by config type (isinstance); no string lookup. Each config class
    (e.g. OVRTXRendererCfg, NewtonWarpRendererCfg) maps to its renderer implementation.

    Raises:
        ValueError: If the config type is not registered.
        ImportError: If the renderer module fails to load.
    """
    cfg_type = type(cfg)
    if cfg_type in _RENDERER_REGISTRY:
        return _RENDERER_REGISTRY[cfg_type]
    if isinstance(cfg, NewtonWarpRendererCfg):
        from .newton_warp_renderer import NewtonWarpRenderer

        _RENDERER_REGISTRY[NewtonWarpRendererCfg] = NewtonWarpRenderer
        return NewtonWarpRenderer
    if isinstance(cfg, OVRTXRendererCfg):
        from .ovrtx_renderer import OVRTXRenderer

        _RENDERER_REGISTRY[OVRTXRendererCfg] = OVRTXRenderer
        return OVRTXRenderer
    raise ValueError(f"No renderer registered for config type '{cfg_type.__name__}'.")

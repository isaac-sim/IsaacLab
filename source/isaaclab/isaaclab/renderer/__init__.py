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

from typing import TYPE_CHECKING, Any

# Import config classes (no circular dependency)
from .newton_warp_renderer_cfg import NewtonWarpRendererCfg

# Import renderer implementations (OVRTXRenderer has no heavy dependencies)
from .ov_rtx_renderer import OVRTXRenderer
from .ov_rtx_renderer_cfg import OVRTXRendererCfg

# Import base classes first
from .renderer import RendererBase
from .renderer_cfg import RendererCfg

# from .kit_app_renderer_cfg import KitAppRendererCfg


if TYPE_CHECKING:
    from typing import Type

    from .newton_warp_renderer import NewtonWarpRenderer

    # from .kit_app_renderer import KitAppRenderer

# Global registry for renderer types (lazy-loaded)
_RENDERER_REGISTRY: dict[str, Any] = {}

__all__ = [
    "RendererBase",
    "RendererCfg",
    "NewtonWarpRendererCfg",
    "NewtonWarpRenderer",
    "OVRTXRendererCfg",
    "OVRTXRenderer",
    "get_renderer_class",
]


# Register only selected renderers to reduce unnecessary imports
def get_renderer_class(name: str) -> type[RendererBase] | None:
    """Get a renderer class by name (lazy-loaded).
    Renderer classes are imported only when requested to avoid loading
    unnecessary dependencies.
    Args:
        name: Renderer type name (e.g., 'newton_warp', 'ov_rtx', 'kit_app').
    Returns:
        Renderer class if found, None otherwise.
    Example:
        >>> renderer_cls = get_renderer_class('newton_warp')
        >>> if renderer_cls:
        >>>     renderer = renderer_cls(cfg)
    """
    # Check if already loaded
    if name in _RENDERER_REGISTRY:
        return _RENDERER_REGISTRY[name]

    # Lazy-load visualizer on first access
    try:
        if name == "newton_warp":
            from .newton_warp_renderer import NewtonWarpRenderer

            _RENDERER_REGISTRY["newton_warp"] = NewtonWarpRenderer
            return NewtonWarpRenderer
        elif name == "ov_rtx":
            from .ov_rtx_renderer import OVRTXRenderer

            _RENDERER_REGISTRY["ov_rtx"] = OVRTXRenderer
            return OVRTXRenderer
        else:
            return None
    except ImportError as e:
        # Log import error but don't crash - renderer just won't be available
        import warnings

        warnings.warn(f"Failed to load renderer '{name}': {e}", ImportWarning)
        return None

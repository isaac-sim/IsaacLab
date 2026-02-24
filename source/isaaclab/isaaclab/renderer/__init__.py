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

Renderer registry and string/config resolution
----------------------------------------------
- **get_renderer_class(name_or_cfg)** accepts either a string or a RendererCfg. When given a config,
  dispatches by type (e.g. isinstance(cfg, IsaacRtxRendererCfg) / NewtonWarpRendererCfg); otherwise
  falls back to name string. Returns Renderer class or None.
- **renderer_cfg_from_type(renderer_type)** maps string → Renderer *config* instance
  ("warp_renderer" → NewtonWarpRendererCfg(), "rtx"/None → IsaacRtxRendererCfg()).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

# Import config classes (no circular dependency)
from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg
from .newton_warp_renderer_cfg import NewtonWarpRendererCfg

# Import base classes first
from .camera_renderer import Renderer as CameraRenderer
from .renderer import RendererBase
from .renderer_cfg import RendererCfg

# from .kit_app_renderer_cfg import KitAppRendererCfg


# from .ov_rtx_renderer_cfg import OVRTXRendererCfg

# Camera-path renderer (TiledCamera inject pattern; lazy to avoid heavy deps at import)
def __getattr__(name: str):
    if name == "NewtonWarpRenderer":
        from .newton_warp_renderer import NewtonWarpRenderer
        return NewtonWarpRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from typing import Type

    from .newton_warp_renderer import NewtonWarpRenderer

    # from .ov_rtx_renderer import OVRTXRenderer
    # from .kit_app_renderer import KitAppRenderer

# Global registry for renderer types (lazy-loaded)
_RENDERER_REGISTRY: dict[str, Any] = {}

__all__ = [
    "CameraRenderer",
    "RendererBase",
    "RendererCfg",
    "IsaacRtxRendererCfg",
    "NewtonWarpRendererCfg",
    "NewtonWarpRenderer",
    "get_renderer_class",
    "renderer_cfg_from_type",
]


def renderer_cfg_from_type(renderer_type: str | None) -> RendererCfg | None:
    """Map Hydra/CLI renderer_type string to a renderer config.

    Used by scene configs so that ``env.scene.base_camera.renderer_type=warp_renderer``
    (or ``=rtx``) still works: set ``base_camera.renderer_cfg = renderer_cfg_from_type(...)``.

    Args:
        renderer_type: ``"warp_renderer"`` -> NewtonWarpRendererCfg();
            ``"rtx"`` or ``None`` -> IsaacRtxRendererCfg() (RTX path).

    Returns:
        NewtonWarpRendererCfg() for ``"warp_renderer"``, IsaacRtxRendererCfg() for ``"rtx"`` or ``None``.
    """
    if renderer_type == "warp_renderer":
        return NewtonWarpRendererCfg()
    return IsaacRtxRendererCfg()  # "rtx" or None -> RTX path


# Register only selected renderers to reduce unnecessary imports
def get_renderer_class(name_or_cfg: Union[str, RendererCfg]) -> type[RendererBase] | None:
    """Get a renderer class by name or by config type.

    When given a RendererCfg, dispatches with isinstance() so we align with IsaacLab
    renderer refactor; when given a string, uses the lazy-loaded name registry.

    Args:
        name_or_cfg: Renderer type name (e.g. 'warp_renderer') or a RendererCfg instance.

    Returns:
        Renderer class if found, None otherwise (e.g. IsaacRtxRendererCfg → None; we use
        built-in RTX path and do not instantiate an IsaacRtxRenderer).
    """
    # Config-based dispatch
    if isinstance(name_or_cfg, RendererCfg):
        if isinstance(name_or_cfg, IsaacRtxRendererCfg):
            return None  # RTX path: no renderer instance, TiledCamera uses Replicator
        if isinstance(name_or_cfg, NewtonWarpRendererCfg):
            from .newton_warp_renderer import NewtonWarpRenderer as _Cls

            return _Cls
        # Unknown config subclass: fall back to renderer_type string
        name_or_cfg = name_or_cfg.renderer_type

    name = name_or_cfg
    # Check if already loaded
    if name in _RENDERER_REGISTRY:
        return _RENDERER_REGISTRY[name]

    # Lazy-load by name
    try:
        if name in ("newton_warp", "warp_renderer"):
            from .newton_warp_renderer import NewtonWarpRenderer as _NewtonWarpRenderer

            _RENDERER_REGISTRY["newton_warp"] = _NewtonWarpRenderer
            _RENDERER_REGISTRY["warp_renderer"] = _NewtonWarpRenderer
            return _NewtonWarpRenderer
        elif name == "ov_rtx":
            from .ov_rtx_renderer import OVRTXRenderer

            _RENDERER_REGISTRY["ov_rtx"] = OVRTXRenderer
            return OVRTXRenderer
        else:
            return None
    except ImportError as e:
        import warnings

        warnings.warn(f"Failed to load renderer '{name}': {e}", ImportWarning)
        return None

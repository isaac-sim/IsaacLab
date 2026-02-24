"""Renderer runtime backends namespace.

This package is intentionally created now as part of rendering-domain consolidation.
"""

from .renderer_cfg import RTXRendererCfg, WarpRendererCfg

__all__ = ["RTXRendererCfg", "WarpRendererCfg"]

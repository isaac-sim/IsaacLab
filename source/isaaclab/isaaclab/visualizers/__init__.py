# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer registry and base interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .visualizer import Visualizer
from .visualizer_cfg import VisualizerCfg

if TYPE_CHECKING:
    from isaaclab_newton.visualizers import NewtonVisualizer, RerunVisualizer
    from isaaclab_physx.visualizers import KitVisualizer

_VISUALIZER_REGISTRY: dict[str, Any] = {}

__all__ = [
    "Visualizer",
    "VisualizerCfg",
    "get_visualizer_class",
]


def get_visualizer_class(name: str) -> type[Visualizer] | None:
    """Get a visualizer class by name (lazy-loaded)."""
    if name in _VISUALIZER_REGISTRY:
        return _VISUALIZER_REGISTRY[name]

    try:
        if name == "newton":
            from isaaclab_newton.visualizers import NewtonVisualizer

            _VISUALIZER_REGISTRY["newton"] = NewtonVisualizer
            return NewtonVisualizer
        if name == "kit":
            from isaaclab_physx.visualizers import KitVisualizer

            _VISUALIZER_REGISTRY["kit"] = KitVisualizer
            return KitVisualizer
        if name == "rerun":
            from isaaclab_newton.visualizers import RerunVisualizer

            _VISUALIZER_REGISTRY["rerun"] = RerunVisualizer
            return RerunVisualizer
        return None
    except ImportError as exc:
        import warnings

        warnings.warn(f"Failed to load visualizer '{name}': {exc}", ImportWarning)
        return None


def __getattr__(name: str) -> Any:
    """Lazily expose backend visualizer config classes from core namespace."""
    if name == "KitVisualizerCfg":
        from isaaclab_physx.visualizers import KitVisualizerCfg

        return KitVisualizerCfg
    if name == "NewtonVisualizerCfg":
        from isaaclab_newton.visualizers import NewtonVisualizerCfg

        return NewtonVisualizerCfg
    if name == "RerunVisualizerCfg":
        from isaaclab_newton.visualizers import RerunVisualizerCfg

        return RerunVisualizerCfg
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

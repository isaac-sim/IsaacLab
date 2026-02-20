# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer Registry.

This module uses a registry pattern to decouple visualizer instantiation
from specific types. Configs can create visualizers via the
`create_visualizer()` factory method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .kit_visualizer_cfg import KitVisualizerCfg
from .newton_visualizer_cfg import NewtonVisualizerCfg
from .rerun_visualizer_cfg import RerunVisualizerCfg
from .visualizer import Visualizer
from .visualizer_cfg import VisualizerCfg

if TYPE_CHECKING:
    from .kit_visualizer import KitVisualizer
    from .newton_visualizer import NewtonVisualizer
    from .rerun_visualizer import RerunVisualizer

_VISUALIZER_REGISTRY: dict[str, Any] = {}

__all__ = [
    "Visualizer",
    "VisualizerCfg",
    "NewtonVisualizerCfg",
    "KitVisualizerCfg",
    "RerunVisualizerCfg",
    "get_visualizer_class",
]


def get_visualizer_class(name: str) -> type[Visualizer] | None:
    """Get a visualizer class by name (lazy-loaded)."""
    if name in _VISUALIZER_REGISTRY:
        return _VISUALIZER_REGISTRY[name]

    try:
        if name == "newton":
            from .newton_visualizer import NewtonVisualizer

            _VISUALIZER_REGISTRY["newton"] = NewtonVisualizer
            return NewtonVisualizer
        if name == "kit":
            from .kit_visualizer import KitVisualizer

            _VISUALIZER_REGISTRY["kit"] = KitVisualizer
            return KitVisualizer
        if name == "rerun":
            from .rerun_visualizer import RerunVisualizer

            _VISUALIZER_REGISTRY["rerun"] = RerunVisualizer
            return RerunVisualizer
        return None
    except ImportError as exc:
        import warnings

        warnings.warn(f"Failed to load visualizer '{name}': {exc}", ImportWarning)
        return None

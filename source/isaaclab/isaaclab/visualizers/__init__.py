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

# Import base classes first
from .visualizer import Visualizer
from .visualizer_cfg import VisualizerCfg

# Global registry for visualizer types (lazy-loaded)
_VISUALIZER_REGISTRY: dict[str, type[Visualizer]] = {}


def get_visualizer_class(name: str) -> type[Visualizer] | None:
    """Get a visualizer class by name (lazy-loaded).

    Args:
        name: Visualizer type name (e.g., 'omniverse').

    Returns:
        Visualizer class if found, None otherwise.
    """
    # Check if already loaded
    if name in _VISUALIZER_REGISTRY:
        return _VISUALIZER_REGISTRY[name]

    # Lazy-load visualizer classes from backend packages
    if name == "omniverse":
        from isaaclab_physx.visualizers import OVVisualizer

        _VISUALIZER_REGISTRY["omniverse"] = OVVisualizer
        return OVVisualizer

    return None


__all__ = [
    "Visualizer",
    "VisualizerCfg",
    "get_visualizer_class",
]

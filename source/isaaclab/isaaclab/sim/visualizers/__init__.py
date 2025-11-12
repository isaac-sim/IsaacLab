# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for visualizer configurations and implementations.

This sub-package contains configuration classes and implementations for different
visualizer backends that can be used with Isaac Lab. The visualizers are used for
debug visualization and monitoring of the simulation, separate from rendering for sensors.

Supported visualizers:
- Newton OpenGL Visualizer: Lightweight OpenGL-based visualizer
- Omniverse Visualizer: High-fidelity Omniverse-based visualizer (coming soon)
- Rerun Visualizer: Web-based visualizer using the rerun library (coming soon)

Visualizer Registry
-------------------
This module uses a registry pattern to decouple visualizer instantiation from specific types.
Visualizer implementations can register themselves using the `register_visualizer` decorator,
and configs can create visualizers via the `create_visualizer()` factory method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Type

# Import base classes first
from .visualizer import Visualizer
from .visualizer_cfg import VisualizerCfg

# Import config classes (no circular dependency)
from .newton_visualizer_cfg import NewtonVisualizerCfg
from .ov_visualizer_cfg import OVVisualizerCfg
from .rerun_visualizer_cfg import RerunVisualizerCfg

# Import visualizer implementations
from .newton_visualizer import NewtonVisualizer

# Global registry for visualizer types (defined after Visualizer import)
_VISUALIZER_REGISTRY: dict[str, Any] = {}

__all__ = [
    "Visualizer",
    "VisualizerCfg",
    "NewtonVisualizer",
    "NewtonVisualizerCfg",
    "OVVisualizerCfg",
    "RerunVisualizerCfg",
    "register_visualizer",
    "get_visualizer_class",
]


def register_visualizer(name: str):
    """Decorator to register a visualizer class with the given name.
    
    This allows visualizer configs to create instances without hard-coded type checking.
    
    Args:
        name: Unique identifier for this visualizer type (e.g., "newton", "rerun", "omniverse").
    
    Example:
        >>> @register_visualizer("newton")
        >>> class NewtonVisualizer(Visualizer):
        >>>     pass
    """

    def decorator(cls: Type[Visualizer]) -> Type[Visualizer]:
        if name in _VISUALIZER_REGISTRY:
            raise ValueError(f"Visualizer '{name}' is already registered!")
        _VISUALIZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_visualizer_class(name: str) -> Type[Visualizer] | None:
    """Get a registered visualizer class by name.
    
    Args:
        name: Visualizer type name.
    
    Returns:
        Visualizer class, or None if not found.
    """
    return _VISUALIZER_REGISTRY.get(name)


# Register built-in visualizers
# Note: Registration happens here to avoid circular imports
_VISUALIZER_REGISTRY["newton"] = NewtonVisualizer



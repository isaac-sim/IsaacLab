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
- Omniverse Visualizer: High-fidelity Omniverse-based visualizer using Isaac Sim viewport
- Rerun Visualizer: Web-based visualizer using the rerun library (coming soon)

Visualizer Registry
-------------------
This module uses a registry pattern to decouple visualizer instantiation from specific types.
Visualizer implementations can register themselves using the `register_visualizer` decorator,
and configs can create visualizers via the `create_visualizer()` factory method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Import base classes first
from .visualizer import Visualizer
from .visualizer_cfg import VisualizerCfg

# Import config classes (no circular dependency)
from .newton_visualizer_cfg import NewtonVisualizerCfg
from .ov_visualizer_cfg import OVVisualizerCfg
from .rerun_visualizer_cfg import RerunVisualizerCfg

# NOTE: Visualizer implementations are NOT imported here to avoid loading
# dependencies for unused visualizers (REQ-10: clean dependency handling).
# They are lazy-loaded via the registry when actually needed.

if TYPE_CHECKING:
    from typing import Type
    from .newton_visualizer import NewtonVisualizer
    from .ov_visualizer import OVVisualizer
    from .rerun_visualizer import RerunVisualizer

# Global registry for visualizer types (lazy-loaded)
_VISUALIZER_REGISTRY: dict[str, Any] = {}

__all__ = [
    "Visualizer",
    "VisualizerCfg",
    "NewtonVisualizerCfg",
    "OVVisualizerCfg",
    "RerunVisualizerCfg",
    "get_visualizer_class",
]

# Note: Visualizer implementation classes (NewtonVisualizer, OVVisualizer, RerunVisualizer)
# are not exported to avoid eager loading. Access them via get_visualizer_class() or
# VisualizerCfg.create_visualizer() instead.


def get_visualizer_class(name: str) -> Type[Visualizer] | None:
    """Get a visualizer class by name (lazy-loaded).
    
    Visualizer classes are imported only when requested to avoid loading
    unnecessary dependencies (REQ-10: clean dependency handling).
    
    Args:
        name: Visualizer type name (e.g., 'newton', 'rerun', 'omniverse', 'ov').
    
    Returns:
        Visualizer class if found, None otherwise.
    
    Example:
        >>> visualizer_cls = get_visualizer_class('newton')
        >>> if visualizer_cls:
        >>>     visualizer = visualizer_cls(cfg)
    """
    # Check if already loaded
    if name in _VISUALIZER_REGISTRY:
        return _VISUALIZER_REGISTRY[name]
    
    # Lazy-load visualizer on first access
    try:
        if name == "newton":
            from .newton_visualizer import NewtonVisualizer
            _VISUALIZER_REGISTRY["newton"] = NewtonVisualizer
            return NewtonVisualizer
        elif name in ("omniverse", "ov"):
            from .ov_visualizer import OVVisualizer
            _VISUALIZER_REGISTRY["omniverse"] = OVVisualizer
            _VISUALIZER_REGISTRY["ov"] = OVVisualizer  # Alias
            return OVVisualizer
        elif name == "rerun":
            from .rerun_visualizer import RerunVisualizer
            _VISUALIZER_REGISTRY["rerun"] = RerunVisualizer
            return RerunVisualizer
        else:
            return None
    except ImportError as e:
        # Log import error but don't crash - visualizer just won't be available
        import warnings
        warnings.warn(f"Failed to load visualizer '{name}': {e}", ImportWarning)
        return None


# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# # Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """Sub-package for visualizer configurations and implementations.

from __future__ import annotations

from typing import TYPE_CHECKING

# Import config classes (no circular dependency)
from .physx_ov_visualizer_cfg import PhysxOVVisualizerCfg

from isaaclab.visualizers import _VISUALIZER_REGISTRY

if TYPE_CHECKING:
    from isaaclab.visualizers import Visualizer

__all__ = [
    "PhysxOVVisualizerCfg",
]


# # Register only selected visualizers to reduce unnecessary imports
def get_visualizer_class(name: str) -> type[Visualizer] | None:
    """Get a visualizer class by name (lazy-loaded).

    Visualizer classes are imported only when requested to avoid loading
    unnecessary dependencies.

    Args:
        name: Visualizer type name (e.g., 'newton', 'rerun', 'omniverse').

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

    if name == "isaacsim_ov":
        from .physx_ov_visualizer_cfg import PhysxOVVisualizerCfg

        _VISUALIZER_REGISTRY["isaacsim_ov"] = PhysxOVVisualizerCfg
        return PhysxOVVisualizerCfg

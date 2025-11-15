# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for visualizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .visualizer import Visualizer


@configclass
class VisualizerCfg:
    """Base configuration for all visualizer backends."""

    visualizer_type: str = "base"
    """Type identifier (e.g., 'newton', 'rerun', 'omniverse')."""

    enabled: bool = False
    """Whether the visualizer is enabled."""

    update_frequency: int = 1
    """Update frequency in simulation steps (1 = every step)."""

    env_indices: list[int] | None = None
    """Environment indices to visualize. None = all environments."""

    enable_markers: bool = True
    """Enable visualization markers (debug drawing)."""

    enable_live_plots: bool = True
    """Enable live plotting of data."""

    camera_position: tuple[float, float, float] = (10.0, 0.0, 3.0)
    """Initial camera position (x, y, z)."""

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target/look-at point (x, y, z)."""

    def get_visualizer_type(self) -> str:
        """Get the visualizer type identifier."""
        return self.visualizer_type

    def create_visualizer(self) -> Visualizer:
        """Create visualizer instance from this config using factory pattern."""
        from . import get_visualizer_class

        visualizer_class = get_visualizer_class(self.visualizer_type)
        if visualizer_class is None:
            raise ValueError(f"Visualizer type '{self.visualizer_type}' is not registered.")
        
        return visualizer_class(self)



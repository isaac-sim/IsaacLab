# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for visualizers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .visualizer import Visualizer


@configclass
class VisualizerCfg:
    """Base configuration for all visualizer backends."""

    visualizer_type: str | None = None
    """Type identifier (e.g., 'newton', 'rerun', 'omniverse')."""

    enable_markers: bool = True
    """Enable visualization markers (debug drawing)."""

    enable_live_plots: bool = True
    """Enable live plotting of data."""

    camera_position: tuple[float, float, float] = (8.0, 8.0, 3.0)
    """Initial camera position (x, y, z) in world coordinates."""

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target/look-at point (x, y, z) in world coordinates."""

    def get_visualizer_type(self) -> str | None:
        """Get the visualizer type identifier."""
        return self.visualizer_type

    def create_visualizer(self) -> "Visualizer":
        """Create visualizer instance from this config using factory pattern."""
        from . import get_visualizer_class

        if self.visualizer_type is None:
            raise ValueError(
                "Cannot create visualizer from base VisualizerCfg class. "
                "Use a specific visualizer config: NewtonVisualizerCfg, RerunVisualizerCfg, or OVVisualizerCfg."
            )

        visualizer_class = get_visualizer_class(self.visualizer_type)
        if visualizer_class is None:
            if self.visualizer_type in ("newton", "rerun"):
                raise ImportError(
                    f"Visualizer '{self.visualizer_type}' requires the Newton Python module and its dependencies. "
                    "Install the Newton backend (e.g., newton package/isaaclab_newton) and retry."
                )
            raise ValueError(
                f"Visualizer type '{self.visualizer_type}' is not registered. "
                "Valid types: 'newton', 'rerun', 'omniverse'."
            )

        return visualizer_class(self)

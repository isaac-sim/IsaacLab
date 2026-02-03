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
    """Base configuration for all visualizer backends.

    Note:
        This is an abstract base class and should not be instantiated directly.
        Use specific visualizer configs like NewtonVisualizerCfg, RerunVisualizerCfg, or OVVisualizerCfg.
    """

    visualizer_type: str | None = None
    """Type identifier (e.g., 'newton', 'rerun', 'omniverse'). Must be overridden by subclasses."""

    # Note: Partial environment visualization will come later
    # env_ids: list[Integer] = []

    enable_markers: bool = True
    """Enable visualization markers (debug drawing)."""

    enable_live_plots: bool = True
    """Enable live plotting of data.

    When set to True for OVVisualizer:
    - Automatically checks the checkboxes for all manager visualizers (Actions, Observations, Rewards, etc.)
    - Keeps the plot frames expanded by default (not collapsed)
    - Makes the live plots visible immediately in the IsaacLab window (docked to the right of the viewport)

    This provides a better out-of-the-box experience when you want to monitor training metrics.
    """

    camera_position: tuple[float, float, float] = (8.0, 8.0, 3.0)
    """Initial camera position (x, y, z) in world coordinates."""

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target/look-at point (x, y, z) in world coordinates."""

    def get_visualizer_type(self) -> str | None:
        """Get the visualizer type identifier.

        Returns:
            The visualizer type string, or None if not set (base class).
        """
        return self.visualizer_type

    def create_visualizer(self) -> Visualizer:
        """Create visualizer instance from this config using factory pattern.

        Raises:
            ValueError: If visualizer_type is None (base class used directly) or not registered.
        """
        from . import get_visualizer_class

        if self.visualizer_type is None:
            raise ValueError(
                "Cannot create visualizer from base VisualizerCfg class. "
                "Use a specific visualizer config: NewtonVisualizerCfg, RerunVisualizerCfg, or OVVisualizerCfg."
            )

        visualizer_class = get_visualizer_class(self.visualizer_type)
        if visualizer_class is None:
            raise ValueError(
                f"Visualizer type '{self.visualizer_type}' is not registered. "
                "Valid types: 'newton', 'rerun', 'omniverse'."
            )

        return visualizer_class(self)

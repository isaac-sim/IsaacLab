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
    """Base configuration for visualizer backends.
    
    This is the base class for all visualizer configurations. Visualizers are used
    for debug visualization and monitoring during simulation, separate from rendering
    for sensors/cameras.
    
    All visualizer backends should inherit from this class and add their specific
    configuration parameters.
    """

    visualizer_type: str = "base"
    """Type identifier for this visualizer (e.g., 'newton', 'rerun', 'omniverse').
    
    This is used by the factory pattern to instantiate the correct visualizer class.
    Subclasses should override this with their specific type identifier.
    """

    enabled: bool = False
    """Whether the visualizer is enabled. Default is False."""

    update_frequency: int = 1
    """Frequency of updates to the visualizer (in simulation steps).
    
    Higher values (e.g., 10) mean the visualizer updates less frequently, improving
    performance at the cost of less responsive visualization. Lower values (e.g., 1)
    provide more responsive visualization but may impact performance.
    Default is 1 (update every step).
    """

    env_indices: list[int] | None = None
    """List of environment indices to visualize. Default is None.
    
    If None, all environments will be visualized. If a list is provided, only the
    specified environments will be visualized. This is useful for reducing the
    visualization overhead when running with many environments.
    
    Example: [0, 1, 2] will visualize only the first 3 environments.
    """

    enable_markers: bool = True
    """Whether to enable visualization markers (debug drawing). Default is True.
    
    Visualization markers are used for debug drawing of points, lines, frames, etc.
    These correspond to the VisualizationMarkers class in isaaclab.markers.
    """

    enable_live_plots: bool = True
    """Whether to enable live plotting of data. Default is True.
    
    Live plots can be used to visualize real-time data such as observations,
    rewards, and other metrics during simulation.
    """

    train_mode: bool = True
    """Whether the visualizer is in training mode (True) or play/inference mode (False).
    
    This affects the UI and controls displayed in the visualizer. In training mode,
    additional controls may be shown for pausing training, adjusting update frequency, etc.
    Default is True.
    """

    camera_position: tuple[float, float, float] = (10.0, 0.0, 3.0)
    """Initial position of the camera in the visualizer. Default is (10.0, 0.0, 3.0)."""

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial target (look-at point) of the camera. Default is (0.0, 0.0, 0.0)."""

    window_width: int = 1920
    """Width of the visualizer window in pixels. Default is 1920."""

    window_height: int = 1080
    """Height of the visualizer window in pixels. Default is 1080."""

    def get_visualizer_type(self) -> str:
        """Get the type of visualizer as a string.
        
        Returns:
            String identifier for the visualizer type.
        """
        return self.visualizer_type

    def create_visualizer(self) -> Visualizer:
        """Factory method to create a visualizer instance from this configuration.
        
        This method uses the visualizer registry to instantiate the appropriate
        visualizer class based on the `visualizer_type` field.
        
        Returns:
            Visualizer instance configured with this config.
        
        Raises:
            ValueError: If the visualizer type is not registered.
        """
        # Import here to avoid circular imports
        from . import get_visualizer_class

        visualizer_class = get_visualizer_class(self.visualizer_type)
        if visualizer_class is None:
            raise ValueError(
                f"Visualizer type '{self.visualizer_type}' is not registered. "
                f"Make sure the visualizer module is imported."
            )
        
        return visualizer_class(self)



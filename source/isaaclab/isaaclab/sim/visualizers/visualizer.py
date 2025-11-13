# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for visualizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider

    from .visualizer_cfg import VisualizerCfg


class Visualizer(ABC):
    """Base class for all visualizer backends.
    
    Visualizers are used for debug visualization and monitoring during simulation,
    separate from rendering for sensors/cameras. Each visualizer backend (Newton OpenGL,
    Omniverse, Rerun) should inherit from this class and implement the required methods.
    
    The visualizer lifecycle follows this pattern:
    1. __init__: Create the visualizer with configuration
    2. initialize: Set up the visualizer with the simulation model/scene
    3. step: Update the visualizer each simulation step
    4. close: Clean up resources when done
    
    Args:
        cfg: Configuration for the visualizer backend.
    """

    def __init__(self, cfg: VisualizerCfg):
        """Initialize the visualizer with configuration.
        
        Args:
            cfg: Configuration for the visualizer backend.
        """
        self.cfg = cfg
        self._is_initialized = False
        self._is_closed = False

    @abstractmethod
    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize the visualizer with the simulation scene.
        
        This method is called once after the simulation scene is created and before
        the simulation starts. It should set up any necessary resources for visualization.
        
        Args:
            scene_data: Optional dictionary containing initial scene data. The contents
                       depend on what's available at initialization time. May include:
                       - "model": Physics model object
                       - "state": Initial physics state
                       - "usd_stage": USD stage
                       The visualizer should handle None or missing keys gracefully.
        """
        pass

    @abstractmethod
    def step(self, dt: float, scene_provider: SceneDataProvider | None = None) -> None:
        """Update the visualizer for one simulation step.
        
        This method is called each simulation step to update the visualization.
        The visualizer should pull any needed data from the scene_provider.
        
        Args:
            dt: Time step in seconds since last visualization update.
            scene_provider: Provider for accessing current scene data (physics state, USD stage, etc.).
                           Visualizers should query this for updated data rather than directly
                           accessing physics managers. May be None if no scene data is available yet.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the visualizer and clean up resources.
        
        This method is called when the simulation is ending or the visualizer
        is no longer needed. It should release any resources held by the visualizer.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the visualizer is still running.
        
        Returns:
            True if the visualizer is running and should continue receiving updates,
            False if it has been closed (e.g., user closed the window).
        """
        pass

    def is_training_paused(self) -> bool:
        """Check if training/simulation is paused by the visualizer.
        
        Some visualizers (like Newton OpenGL) provide controls to pause the simulation
        while keeping the visualizer running. This is useful for debugging.
        
        Returns:
            True if the user has paused training/simulation, False otherwise.
            Default implementation returns False (no pause support).
        """
        return False

    def is_rendering_paused(self) -> bool:
        """Check if rendering is paused by the visualizer.
        
        Some visualizers allow pausing rendering while keeping simulation running,
        which can improve performance during training.
        
        Returns:
            True if rendering is paused, False otherwise.
            Default implementation returns False (no pause support).
        """
        return False

    @property
    def is_initialized(self) -> bool:
        """Check if the visualizer has been initialized.
        
        Returns:
            True if initialize() has been called successfully.
        """
        return self._is_initialized

    @property
    def is_closed(self) -> bool:
        """Check if the visualizer has been closed.
        
        Returns:
            True if close() has been called.
        """
        return self._is_closed
    
    def supports_markers(self) -> bool:
        """Check if this visualizer supports visualization markers.
        
        Visualization markers are geometric shapes (spheres, arrows, frames, etc.) 
        used for debug visualization. They are typically managed by the scene/environment
        but rendered by the visualizer.
        
        Returns:
            True if the visualizer can display VisualizationMarkers, False otherwise.
            Default implementation returns False.
        """
        return False
    
    def supports_live_plots(self) -> bool:
        """Check if this visualizer supports live plots.
        
        Live plots display time-series data (observations, rewards, etc.) in real-time
        via UI widgets. They are typically managed by manager-based environments.
        
        Returns:
            True if the visualizer can display live plots, False otherwise.
            Default implementation returns False.
        """
        return False


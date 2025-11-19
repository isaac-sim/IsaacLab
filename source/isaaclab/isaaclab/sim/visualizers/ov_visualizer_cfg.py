# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Omniverse-based visualizer."""

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class OVVisualizerCfg(VisualizerCfg):
    """Configuration for Omniverse visualizer using Isaac Sim viewport.
    
    Displays USD stage, VisualizationMarkers, and LivePlots.
    Can attach to existing app or launch standalone.
    """
    
    visualizer_type: str = "omniverse"
    """Type identifier for Omniverse visualizer."""
    
    viewport_name: str | None = "Visualizer"
    """Viewport name to use. If None, uses active viewport."""
    
    create_viewport: bool = True #False
    """Create new viewport with specified name and camera pose."""
    
    dock_position: str = "SAME"
    """Dock position for new viewport. Options: 'LEFT', 'RIGHT', 'BOTTOM', 'SAME' (tabs with existing)."""
    
    window_width: int = 777 # 1920 
    """Viewport width in pixels."""
    
    window_height: int = 777 # 1080
    """Viewport height in pixels."""
    
    camera_position: tuple[float, float, float] = (10.0, 10.0, 3.0)
    """Initial camera position (x, y, z)."""
    
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target/look-at point (x, y, z)."""

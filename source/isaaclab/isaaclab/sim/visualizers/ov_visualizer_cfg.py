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
    
    viewport_name: str | None = "/OmniverseKit/Viewport"
    """Viewport name to use. If None, uses active viewport."""
    
    create_viewport: bool = False
    """Create new viewport with specified name and camera pose."""
    
    window_width: int = 1920
    """Viewport width in pixels."""
    
    window_height: int = 1080
    """Viewport height in pixels."""
    
    launch_app_if_missing: bool = True
    """Launch Isaac Sim if not already running."""
    
    app_experience: str = "isaac-sim.python.kit"
    """Isaac Sim experience file for standalone launch."""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Omniverse-based visualizer."""

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class OVVisualizerCfg(VisualizerCfg):
    """Configuration for Omniverse-based visualizer.
    
    This visualizer uses the Isaac Sim application viewport for visualization.
    It automatically displays:
    - The USD stage (all environment prims)
    - VisualizationMarkers (via USD prims)
    - LivePlots (via Isaac Lab UI widgets)
    
    The visualizer can operate in two modes:
    1. Attached mode: Uses an existing Isaac Sim app instance
    2. Standalone mode: Launches a new Isaac Sim app if none exists
    """
    
    visualizer_type: str = "omniverse"
    
    # Viewport settings
    viewport_name: str = "/OmniverseKit/Viewport"
    """Name of the viewport to use. If None, uses the default active viewport."""
    
    camera_position: tuple[float, float, float] | None = (10.0, 10.0, 10.0)
    """Initial camera position for viewport (x, y, z). If None, keeps current camera pose."""
    
    camera_target: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    """Initial camera target/look-at point (x, y, z). If None, keeps current target."""
    
    # App launch settings (for standalone mode)
    launch_app_if_missing: bool = True
    """If True and no app is running, launch a new Isaac Sim app instance."""
    
    app_experience: str = "isaac-sim.python.kit"
    """Isaac Sim app experience file to use when launching standalone app."""
    
    # Environment visibility (for partial visualization - future use)
    # NOTE: Partial visualization (REQ-11) is not implemented in this minimal version
    # visualize_all_envs: bool = True
    # env_indices_to_visualize: list[int] | None = None

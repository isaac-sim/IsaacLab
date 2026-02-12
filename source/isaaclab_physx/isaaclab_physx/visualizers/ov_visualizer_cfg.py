# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for Omniverse-based visualizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

if TYPE_CHECKING:
    from .ov_visualizer import RenderMode


@configclass
class OVVisualizerCfg(VisualizerCfg):
    """Configuration for Omniverse visualizer using Isaac Sim viewport.

    Displays USD stage, VisualizationMarkers, and LivePlots.
    Can attach to existing app or launch standalone.
    """

    visualizer_type: str = "omniverse"
    """Type identifier for Omniverse visualizer."""

    render_throttle_period: int = 5
    """Throttle period for rendering updates.

    This controls how frequently UI elements are updated when in
    NO_RENDERING mode. A higher value means less frequent UI updates,
    improving performance.
    """

    default_render_mode: RenderMode | None = None
    """Default render mode. If None, auto-detected based on settings."""

    viewport_name: str | None = "Viewport"
    """Viewport name to use. If None, uses active viewport."""

    create_viewport: bool = False
    """Create new viewport with specified name and camera pose."""

    dock_position: str = "SAME"
    """Dock position for new viewport.

    Options: 'LEFT', 'RIGHT', 'BOTTOM', 'SAME' (tabs with existing).
    """

    window_width: int = 1280
    """Viewport width in pixels."""

    window_height: int = 720
    """Viewport height in pixels."""

    camera_prim_path: str = "/OmniverseKit_Persp"
    """Path to the camera prim for viewport rendering."""

    camera_position: tuple[float, float, float] = (2.5, 2.5, 2.5)
    """Initial camera eye position."""

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look-at target."""

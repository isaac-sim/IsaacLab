# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Kit-based visualizer."""

from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


@configclass
class KitVisualizerCfg(VisualizerCfg):
    """Configuration for Kit visualizer using Isaac Sim viewport."""

    visualizer_type: str = "kit"
    """Type identifier for Kit visualizer."""

    viewport_name: str = "Visualizer Viewport"
    """Viewport name to use when :attr:`create_viewport` is True."""

    create_viewport: bool = True
    """Create new viewport with specified name and camera pose."""

    visualizer_camera_prim_path: str = "/World/Cameras/KitVisualizerCamera"
    """Dedicated camera prim path controlled by the Kit visualizer."""

    focal_length: float = 12.0
    """Focal length in millimeters applied to the dedicated visualizer camera."""

    enable_visualizer_cam: bool = True
    """Whether the Kit visualizer should control/bind a dedicated viewport camera.

    If False, Kit does not create/switch camera prims and ignores visualizer camera control
    updates (including eye/lookat and cam_source handling).
    """

    headless: bool = False
    """Run without creating viewport windows when supported by the app."""

    dock_position: str = "SAME"
    """Dock position for new viewport. Options: 'LEFT', 'RIGHT', 'BOTTOM', 'SAME'."""

    window_width: int = 1280
    """Viewport width in pixels."""

    window_height: int = 720
    """Viewport height in pixels."""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton OpenGL Visualizer."""

from typing import Literal

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class NewtonVisualizerCfg(VisualizerCfg):
    """Configuration for Newton OpenGL visualizer.
    
    Lightweight OpenGL-based visualizer with real-time 3D rendering, interactive
    camera controls, and debug visualization (contacts, joints, springs, COM).
    
    Requires: pyglet >= 2.1.6, imgui_bundle >= 1.92.0
    """

    visualizer_type: str = "newton"
    """Type identifier for Newton visualizer."""

    window_width: int = 1920
    """Window width in pixels."""

    window_height: int = 1080
    """Window height in pixels."""

    camera_position: tuple[float, float, float] = (10.0, 0.0, 3.0)
    """Initial camera position (x, y, z)."""

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target/look-at point (x, y, z)."""

    # Newton-specific settings
    fps: int = 60
    """Target FPS."""

    show_joints: bool = True
    """Show joint visualizations."""

    show_contacts: bool = False
    """Show contact visualizations."""

    show_springs: bool = False
    """Show spring visualizations."""

    show_com: bool = False
    """Show center of mass visualizations."""

    show_ui: bool = True
    """Show UI sidebar (toggle with 'H' key)."""

    enable_shadows: bool = True
    """Enable shadow rendering."""

    enable_sky: bool = True
    """Enable sky rendering."""

    enable_wireframe: bool = False
    """Enable wireframe rendering mode."""

    up_axis: Literal["X", "Y", "Z"] = "Z"
    """Up axis for visualizer (should match simulation)."""

    fov: float = 60.0
    """Camera field of view in degrees."""

    near_plane: float = 0.1
    """Camera near clipping plane distance."""

    far_plane: float = 1000.0
    """Camera far clipping plane distance."""

    background_color: tuple[float, float, float] = (0.53, 0.81, 0.92)
    """Background/sky color RGB [0,1] (light blue)."""

    ground_color: tuple[float, float, float] = (0.18, 0.20, 0.25)
    """Ground color RGB [0,1] (dark gray)."""

    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Light color RGB [0,1] (white)."""

    enable_pause_training: bool = True
    """Enable pause training button in UI."""

    enable_pause_rendering: bool = True
    """Enable pause rendering button in UI."""

    show_training_controls: bool = True
    """Show Isaac Lab training controls in UI."""

    render_mode: Literal["rgb", "depth", "collision"] = "rgb"
    """Rendering mode: rgb (standard), depth, or collision."""



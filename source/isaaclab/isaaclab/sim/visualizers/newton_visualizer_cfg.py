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
    """Configuration for Newton OpenGL Visualizer.
    
    The Newton OpenGL Visualizer is a lightweight, fast visualizer built on OpenGL.
    It is designed for interactive real-time visualization of simulations with minimal
    performance overhead. It requires pyglet (version >= 2.1.6) and imgui_bundle
    (version >= 1.92.0) to be installed.
    
    Features:
    - Real-time 3D visualization
    - Interactive camera controls
    - Debug visualization (contacts, joints, springs, COM)
    - Training controls (pause training, pause rendering, update frequency)
    - Lightweight and fast
    
    Note:
        The Newton Visualizer currently only supports visualization of collision shapes,
        not visual shapes.
    """

    # Visualizer type identifier
    visualizer_type: str = "newton"
    """Type identifier for Newton visualizer. Used by the factory pattern."""

    # Override defaults for Newton visualizer
    camera_position: tuple[float, float, float] = (10.0, 0.0, 3.0)
    """Initial position of the camera. Default is (10.0, 0.0, 3.0)."""

    window_width: int = 1920
    """Width of the visualizer window in pixels. Default is 1920."""

    window_height: int = 1080
    """Height of the visualizer window in pixels. Default is 1080."""

    # Newton-specific settings
    fps: int = 60
    """Target frames per second for the visualizer. Default is 60."""

    show_joints: bool = True
    """Whether to show joint visualizations. Default is True."""

    show_contacts: bool = False
    """Whether to show contact visualizations. Default is False."""

    show_springs: bool = False
    """Whether to show spring visualizations. Default is False."""

    show_com: bool = False
    """Whether to show center of mass visualizations. Default is False."""

    show_ui: bool = True
    """Whether to show the UI sidebar. Default is True.
    
    The UI can be toggled with the 'H' key during runtime.
    """

    enable_shadows: bool = True
    """Whether to enable shadow rendering. Default is True."""

    enable_sky: bool = True
    """Whether to enable sky rendering. Default is True."""

    enable_wireframe: bool = False
    """Whether to enable wireframe rendering mode. Default is False."""

    up_axis: Literal["X", "Y", "Z"] = "Z"
    """The up axis for the visualizer. Default is "Z".
    
    This should typically match the up axis of your simulation environment.
    """

    fov: float = 60.0
    """Field of view for the camera in degrees. Default is 60.0."""

    near_plane: float = 0.1
    """Near clipping plane distance for the camera. Default is 0.1."""

    far_plane: float = 1000.0
    """Far clipping plane distance for the camera. Default is 1000.0."""

    background_color: tuple[float, float, float] = (0.53, 0.81, 0.92)
    """Background color (sky color) as RGB values in range [0, 1]. 
    Default is (0.53, 0.81, 0.92) (light blue)."""

    ground_color: tuple[float, float, float] = (0.18, 0.20, 0.25)
    """Ground color as RGB values in range [0, 1]. 
    Default is (0.18, 0.20, 0.25) (dark gray)."""

    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Light color as RGB values in range [0, 1]. Default is (1.0, 1.0, 1.0) (white)."""

    enable_pause_training: bool = True
    """Whether to enable the pause training button in the UI. Default is True.
    
    When enabled, users can pause the simulation/training while keeping the
    visualizer running, which is useful for debugging.
    """

    enable_pause_rendering: bool = True
    """Whether to enable the pause rendering button in the UI. Default is True.
    
    When enabled, users can pause rendering while keeping simulation/training
    running, which can improve training performance.
    """

    show_training_controls: bool = True
    """Whether to show IsaacLab-specific training controls in the UI. Default is True.
    
    This includes controls for pausing training, pausing rendering, and adjusting
    the visualizer update frequency.
    """

    render_mode: Literal["rgb", "depth", "collision"] = "rgb"
    """Rendering mode for the visualizer. Default is "rgb".
    
    - "rgb": Standard RGB rendering
    - "depth": Depth visualization
    - "collision": Show collision shapes only
    """



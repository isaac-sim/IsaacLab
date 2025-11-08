# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Omniverse Visualizer."""

from typing import Literal

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class OVVisualizerCfg(VisualizerCfg):
    """Configuration for Omniverse Visualizer.
    
    The Omniverse Visualizer uses the Omniverse SDK to provide high-fidelity
    visualization of the simulation. This is currently implemented through the
    Isaac Sim app, but will eventually use the standalone Omniverse SDK module.
    
    Features:
    - High-fidelity rendering with RTX
    - Full visual shape support (not just collision shapes)
    - USD-based scene representation
    - Advanced lighting and materials
    - Integration with Omniverse ecosystem
    
    Note:
        The Omniverse Visualizer has higher overhead than the Newton Visualizer
        and requires Omniverse/Isaac Sim to be installed.
    """

    # Override defaults for Omniverse visualizer
    camera_position: tuple[float, float, float] = (10.0, 10.0, 10.0)
    """Initial position of the camera. Default is (10.0, 10.0, 10.0)."""

    window_width: int = 1920
    """Width of the visualizer window in pixels. Default is 1920."""

    window_height: int = 1080
    """Height of the visualizer window in pixels. Default is 1080."""

    # Omniverse-specific settings
    viewport_name: str = "/OmniverseKit_Persp"
    """Name of the viewport to use for visualization. Default is "/OmniverseKit_Persp"."""

    show_origin_axis: bool = True
    """Whether to show the origin axis. Default is True."""

    show_grid: bool = True
    """Whether to show the grid. Default is True."""

    grid_scale: float = 1.0
    """Scale of the grid. Default is 1.0."""

    enable_scene_lights: bool = True
    """Whether to enable scene lights. Default is True."""

    default_light_intensity: float = 3000.0
    """Default intensity for scene lights. Default is 3000.0."""

    enable_dome_light: bool = True
    """Whether to enable dome (environment) lighting. Default is True."""

    dome_light_intensity: float = 1000.0
    """Intensity of the dome light. Default is 1000.0."""

    dome_light_texture: str | None = None
    """Path to HDR texture for dome light. Default is None (use default).
    
    If specified, should be a path to an HDR image file for image-based lighting.
    """

    camera_projection: Literal["perspective", "orthographic"] = "perspective"
    """Camera projection type. Default is "perspective"."""

    fov: float = 60.0
    """Field of view for the camera in degrees (for perspective projection). Default is 60.0."""

    near_plane: float = 0.1
    """Near clipping plane distance. Default is 0.1."""

    far_plane: float = 10000.0
    """Far clipping plane distance. Default is 10000.0."""

    enable_ui: bool = True
    """Whether to enable the Omniverse UI. Default is True.
    
    When disabled, runs in a more minimal mode which can improve performance.
    """

    ui_window_layout: str | None = None
    """Custom UI window layout file. Default is None (use default layout).
    
    Can be a path to a .json file specifying the UI layout.
    """

    show_selection_outline: bool = True
    """Whether to show selection outline on picked objects. Default is True."""

    show_physics_debug_viz: bool = False
    """Whether to show physics debug visualization (contacts, joints). Default is False."""

    show_bounding_boxes: bool = False
    """Whether to show bounding boxes. Default is False."""

    display_options: int = 3094
    """Display options bitmask. Default is 3094.
    
    This controls what is visible in the stage. The default value (3094) hides
    extra objects that shouldn't appear in visualization. Another common value
    is 3286 for the regular editor experience.
    """

    enable_live_sync: bool = False
    """Whether to enable live sync with Omniverse. Default is False.
    
    When enabled, allows other Omniverse clients to connect and view the simulation.
    """

    antialiasing: Literal["Off", "FXAA", "TAA", "DLSS", "DLAA"] = "TAA"
    """Anti-aliasing mode. Default is "TAA".
    
    - Off: No anti-aliasing
    - FXAA: Fast approximate anti-aliasing
    - TAA: Temporal anti-aliasing
    - DLSS: NVIDIA Deep Learning Super Sampling (requires RTX GPU)
    - DLAA: NVIDIA Deep Learning Anti-Aliasing (requires RTX GPU)
    """

    enable_post_processing: bool = True
    """Whether to enable post-processing effects. Default is True."""

    enable_motion_blur: bool = False
    """Whether to enable motion blur. Default is False."""

    enable_depth_of_field: bool = False
    """Whether to enable depth of field. Default is False."""

    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Background color as RGB values in range [0, 1]. Default is (0.0, 0.0, 0.0) (black)."""

    capture_on_play: bool = False
    """Whether to start capturing when play is pressed. Default is False.
    
    Useful for recording the simulation for later playback.
    """

    capture_path: str | None = None
    """Path to save captures. Default is None (don't capture).
    
    When specified, frames will be captured to this path.
    """

    capture_format: Literal["png", "jpg", "exr"] = "png"
    """Format for captured images. Default is "png"."""



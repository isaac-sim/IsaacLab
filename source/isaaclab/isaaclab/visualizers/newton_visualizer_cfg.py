# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton OpenGL Visualizer."""

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class NewtonVisualizerCfg(VisualizerCfg):
    """Configuration for Newton OpenGL visualizer."""

    visualizer_type: str = "newton"
    """Type identifier for Newton visualizer."""

    window_width: int = 1920
    """Window width in pixels."""

    window_height: int = 1080
    """Window height in pixels."""

    update_frequency: int = 1
    """Visualizer update frequency (updates every N frames)."""

    show_joints: bool = False
    """Show joint visualization."""

    show_contacts: bool = False
    """Show contact visualization."""

    show_springs: bool = False
    """Show spring visualization."""

    show_com: bool = False
    """Show center of mass visualization."""

    enable_shadows: bool = True
    """Enable shadow rendering."""

    enable_sky: bool = True
    """Enable sky rendering."""

    enable_wireframe: bool = False
    """Enable wireframe rendering."""

    sky_upper_color: tuple[float, float, float] = (0.2, 0.4, 0.6)
    """Sky upper color RGB [0,1]."""

    sky_lower_color: tuple[float, float, float] = (0.5, 0.6, 0.7)
    """Sky lower color RGB [0,1]."""

    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Light color RGB [0,1]."""

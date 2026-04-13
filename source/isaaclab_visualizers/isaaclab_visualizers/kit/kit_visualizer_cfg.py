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

    viewport_name: str | None = "Visualizer Viewport"
    """Viewport name to use. If None, uses active viewport."""

    create_viewport: bool = False
    """Create new viewport with specified name and camera pose."""

    headless: bool = False
    """Run without creating viewport windows when supported by the app."""

    dock_position: str = "SAME"
    """Dock position for new viewport. Options: 'LEFT', 'RIGHT', 'BOTTOM', 'SAME'."""

    window_width: int = 1280
    """Viewport width in pixels."""

    window_height: int = 720
    """Viewport height in pixels."""

    rendering_mode: str | None = "performance"
    """Kit/RTX profile name from :attr:`~isaaclab.sim.SimulationCfg.rendering_mode_cfgs` (default ``performance``).

    Built-in ``performance`` / ``balanced`` / ``quality`` baselines are defined in
    :mod:`isaaclab_physx.rendering.rtx_rendering_mode_presets` and applied through the same mechanism as RTX camera
    renderers; no separate copy lives under ``isaaclab_visualizers``.

    Set to ``None`` to skip applying a named profile (USD / Kit defaults).
    """

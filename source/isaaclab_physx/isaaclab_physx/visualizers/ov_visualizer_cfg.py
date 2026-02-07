# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Omniverse visualizer in PhysX-based SimulationContext."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from .ov_visualizer import RenderMode

if TYPE_CHECKING:
    from .ov_visualizer import OVVisualizer


@configclass
class OVVisualizerCfg(VisualizerCfg):
    """Configuration for Omniverse visualizer in PhysX-based SimulationContext.

    This configuration extends :class:`VisualizerCfg` and is used by the
    :class:`OVVisualizer` class which manages viewport/rendering for
    PhysX-based SimulationContext workflows.
    """

    visualizer_type: str = "omniverse"
    """Type identifier for Omniverse visualizer."""

    default_render_mode: RenderMode | None = None
    """Default render mode for the visualizer.

    If None, the render mode will be automatically determined based on GUI availability:
    - NO_GUI_OR_RENDERING: When no GUI and offscreen rendering is disabled
    - PARTIAL_RENDERING: When no GUI but offscreen rendering is enabled
    - FULL_RENDERING: When GUI is available (local, livestreamed, or XR)

    See :class:`RenderMode` for available options.
    """

    render_throttle_period: int = 5
    """Throttle period for rendering updates.

    This controls how frequently UI elements are updated when in NO_RENDERING mode.
    A higher value means less frequent UI updates, improving performance.
    """

    camera_prim_path: str = "/OmniverseKit_Persp"
    """Path to the camera primitive in the USD stage."""

    warmup_renders: int = 2
    """Number of warmup renders to perform on hard reset.

    This is used to initialize replicator buffers before the simulation starts.
    """

    viewport_name: str | None = "Viewport"
    """Viewport name to use. If None, uses active viewport."""

    create_viewport: bool = False
    """Create new viewport with specified name and camera pose."""

    dock_position: str = "SAME"
    """Dock position for new viewport. Options: 'LEFT', 'RIGHT', 'BOTTOM', 'SAME' (tabs with existing)."""

    window_width: int = 1280
    """Viewport width in pixels."""

    window_height: int = 720
    """Viewport height in pixels."""

    def create_visualizer(self) -> OVVisualizer:
        """Create OVVisualizer instance from this config.

        Returns:
            OVVisualizer instance configured with this config.
        """
        from .ov_visualizer import OVVisualizer

        return OVVisualizer(self)


# Backward compatibility alias
PhysxOVVisualizerCfg = OVVisualizerCfg

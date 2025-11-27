# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rerun visualizer."""

from __future__ import annotations

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class RerunVisualizerCfg(VisualizerCfg):
    """Configuration for Rerun visualizer (web-based visualization).

    Provides time scrubbing, 3D navigation, data filtering, and .rrd recording.
    Requires Newton physics backend and rerun-sdk: `pip install rerun-sdk`
    """

    visualizer_type: str = "rerun"
    """Type identifier for Rerun visualizer."""

    server_mode: bool = True
    """Run Rerun in server mode (gRPC for web viewer)."""

    server_address: str = "127.0.0.1:9876"
    """Server address and port for gRPC mode."""

    launch_viewer: bool = True
    """Auto-launch web viewer in browser."""

    app_id: str = "isaaclab-simulation"
    """Application identifier shown in viewer title."""

    keep_historical_data: bool = False
    """Keep transform history for time scrubbing (False = constant memory for training)."""

    keep_scalar_history: bool = True
    """Keep scalar/plot history in timeline."""

    record_to_rrd: str | None = None
    """Path to save .rrd recording file. None = no recording."""

    camera_position: tuple[float, float, float] = (5.0, 5.0, 2.0)
    """Initial camera position (x, y, z). Closer to robots than default (10, 10, 3)."""

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target/look-at point (x, y, z)."""

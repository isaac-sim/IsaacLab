# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rerun visualizer."""

from __future__ import annotations

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class RerunVisualizerCfg(VisualizerCfg):
    """Configuration for Rerun visualizer (web-based visualization)."""

    visualizer_type: str = "rerun"
    """Type identifier for Rerun visualizer."""

    app_id: str = "isaaclab-simulation"
    """Application identifier shown in viewer title."""

    web_port: int = 9090
    """Port of the local rerun web viewer which is launched in the browser."""

    grpc_port: int = 9876
    """Port of the rerun gRPC server (used when serving web viewer externally)."""

    bind_address: str | None = "0.0.0.0"
    """If set, start a rerun server bound to this address (e.g. '0.0.0.0') and connect to it."""

    open_browser: bool = True
    """Whether to auto-open a browser when serving the rerun web viewer."""

    keep_historical_data: bool = False
    """Keep transform history for time scrubbing (False = constant memory for training)."""

    keep_scalar_history: bool = False
    """Keep scalar/plot history in timeline."""

    record_to_rrd: str | None = None
    """Path to save .rrd recording file. None = no recording."""

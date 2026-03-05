# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rerun visualizer."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


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
    """Bind host used when starting a rerun server and for display endpoint formatting.

    Notes:
    - If an existing rerun server is already reachable on ``grpc_port``, it is reused.
    - Local browser links normalize common loopback/wildcard hosts to ``127.0.0.1``.
    """

    open_browser: bool = True
    """Whether to attempt opening the rerun web viewer URL in a browser."""

    auto_kill_stale_rerun_process: bool = True
    """Whether to terminate a stale rerun process blocking the web port before spawning."""

    keep_historical_data: bool = False
    """Keep transform history for time scrubbing (False = constant memory for training)."""

    keep_scalar_history: bool = False
    """Keep scalar/plot history in timeline."""

    record_to_rrd: str | None = None
    """Path to save .rrd recording file. None = no recording."""

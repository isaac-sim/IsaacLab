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

    recording_output_path: str | None = None
    """Recording output path.

    Interpretation:
    - ``None``: no recording.
    - path ending with ``.rrd``: explicit file path (single recording mode), or basename seed for multi-window mode.
    - any other path: directory for generated recording files.
    """

    record_every_n_iterations: int | None = None
    """Start a recording window every N training iterations.

    If ``None`` or <= 0, periodic recording windows are disabled.
    """

    record_window_length_s: float | None = None
    """Length of each periodic recording window in simulation seconds.

    If ``None`` and periodic mode is enabled, a recording window stays active until shutdown.
    """

    keep_last_n_recordings: int | None = None
    """If set, keep only the newest N generated recording files and delete older ones."""

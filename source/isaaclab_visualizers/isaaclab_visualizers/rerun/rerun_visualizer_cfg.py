# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rerun visualizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.utils import config_field
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

if TYPE_CHECKING:
    from .rerun_visualizer import RerunVisualizer


@dataclass
class RerunVisualizerCfg(VisualizerCfg):
    """Configuration for Rerun visualizer (web-based visualization)."""

    class_type: type[RerunVisualizer] | str = config_field("{DIR}.rerun_visualizer:RerunVisualizer")
    """Visualizer implementation class."""

    visualizer_type: str = config_field("rerun")
    """Type identifier for Rerun visualizer."""

    app_id: str = config_field("isaaclab-simulation")
    """Application identifier shown in viewer title."""

    web_port: int = config_field(9090)
    """Port of the local rerun web viewer whose URL is logged during initialization."""

    grpc_port: int = config_field(9876)
    """Port of the rerun gRPC server (used when serving web viewer externally)."""

    bind_address: str | None = config_field("0.0.0.0")
    """Host used for endpoint formatting and reuse checks.

    Notes:
    - If an existing rerun server is reachable on ``grpc_port``, it is reused.
    - New server startup is managed by ``newton.viewer.ViewerRerun`` via the rerun Python SDK.
    - Local browser links normalize common loopback/wildcard hosts to ``127.0.0.1``.
    """

    open_browser: bool = config_field(False)
    """Whether to attempt opening the rerun web viewer URL in a browser.

    The viewer URL is always logged during initialization. Set this to ``True`` to auto-launch it.
    """

    keep_historical_data: bool = config_field(False)
    """Keep transform history for time scrubbing (False = constant memory for training)."""

    keep_scalar_history: bool = config_field(False)
    """Accumulate scalars as a time-series in the Rerun timeline (True = live plot history, False = constant memory).

    When :attr:`~isaaclab.visualizers.VisualizerCfg.enable_live_plots` is ``True`` (the default),
    this is automatically forced to ``True`` so that scalar values accumulate as a time series in
    the Rerun viewer.  Set to ``False`` explicitly to reduce memory usage when scalar history is
    not needed, but note this will disable live plot curves.
    """

    show_particles: bool = config_field(True)
    """Whether to show model particles.

    Disable this option to reduce streaming overhead for large particle clouds.
    """

    record_to_rrd: str | None = config_field(None)
    """Path to save .rrd recording file. None = no recording."""

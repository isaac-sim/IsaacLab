# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Viser visualizer."""

from __future__ import annotations

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class ViserVisualizerCfg(VisualizerCfg):
    """Configuration for Viser visualizer (web-based visualization)."""

    visualizer_type: str = "viser"
    """Type identifier for Viser visualizer."""

    port: int = 8080
    """Port of the local viser web server."""

    label: str | None = "Isaac Lab Simulation"
    """Optional label shown in the viewer page title."""

    verbose: bool = True
    """Whether to print viewer server startup information."""

    share: bool = False
    """Whether to request a public share URL from viser."""

    record_to_viser: str | None = None
    """Path to save a .viser recording file. None = no recording."""

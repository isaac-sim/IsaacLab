# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse Visualizer implementation."""

from __future__ import annotations

from .ov_visualizer_cfg import OVVisualizerCfg
from .visualizer import Visualizer


class OmniverseVisualizer(Visualizer):
    """Omniverse Visualizer implementation."""
    def __init__(self, cfg: OVVisualizerCfg):
        super().__init__(cfg)
        # stub for now
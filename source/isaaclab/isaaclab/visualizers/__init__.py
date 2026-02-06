# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Visualizer Registry
# -------------------
# This module uses a registry pattern to decouple visualizer instantiation from specific types.
# Visualizer implementations can register themselves using the `register_visualizer` decorator,
# and configs can create visualizers via the `create_visualizer()` factory method.
# """

from __future__ import annotations

from typing import Any

# Import base classes first
from .visualizer import Visualizer
from .visualizer_cfg import VisualizerCfg

# Global registry for visualizer types (lazy-loaded)
_VISUALIZER_REGISTRY: dict[str, Any] = {}

__all__ = [
    "Visualizer",
    "VisualizerCfg",
]

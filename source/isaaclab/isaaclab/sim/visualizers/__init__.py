# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for visualizer configurations and implementations.

This sub-package contains configuration classes and implementations for different
visualizer backends that can be used with Isaac Lab. The visualizers are used for
debug visualization and monitoring of the simulation, separate from rendering for sensors.

Supported visualizers:
- Newton OpenGL Visualizer: Lightweight OpenGL-based visualizer
- Omniverse Visualizer: High-fidelity Omniverse-based visualizer (coming soon)
- Rerun Visualizer: Web-based visualizer using the rerun library (coming soon)
"""

from .visualizer import Visualizer
from .visualizer_cfg import VisualizerCfg
from .newton_visualizer import NewtonVisualizer
from .newton_visualizer_cfg import NewtonVisualizerCfg
from .ov_visualizer_cfg import OVVisualizerCfg
from .rerun_visualizer_cfg import RerunVisualizerCfg

__all__ = [
    "Visualizer",
    "VisualizerCfg",
    "NewtonVisualizer",
    "NewtonVisualizerCfg",
    "OVVisualizerCfg",
    "RerunVisualizerCfg",
]



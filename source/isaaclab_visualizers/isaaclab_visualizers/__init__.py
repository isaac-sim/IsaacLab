# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer backends for Isaac Lab.

Concrete visualizer configs carry their implementation in ``class_type``, which is resolved lazily
when the visualizer is constructed. Import a specific backend only when needed:

  from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
  from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonGLVisualizerCfg
  from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg
  from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("isaaclab_visualizers")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer backends for Isaac Lab.

Visualizers are loaded lazily by type (kit, newton, rerun, viser) via the factory in
isaaclab.visualizers. Import a specific backend only when needed:

  from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
  from isaaclab_visualizers.newton import NewtonVisualizer, NewtonVisualizerCfg
  from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg
  from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg
"""

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated: ViewportCameraController compatibility shim.

Camera tracking is now built into :class:`~isaaclab_visualizers.kit.KitVisualizer`.
Configure ``eye``, ``lookat``, ``origin_type``, and ``origin_track_path`` directly on
:class:`~isaaclab_visualizers.kit.KitVisualizerCfg` and add it to
:attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ViewerCfg


class ViewportCameraController:
    """Deprecated compatibility shim for the removed ``ViewportCameraController`` class.

    .. deprecated::
        :class:`ViewportCameraController` has been removed. Camera tracking is now built into
        :class:`~isaaclab_visualizers.kit.KitVisualizer`. Set ``origin_type``,
        ``origin_track_path``, ``eye``, and ``lookat`` directly on
        :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` and pass it to
        :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs`.
    """

    def __init__(self, env: object, cfg: ViewerCfg):
        warnings.warn(
            "ViewportCameraController is deprecated and has been removed. "
            "Camera tracking is now built into KitVisualizer — configure "
            "origin_type, origin_track_path, eye, and lookat directly on KitVisualizerCfg "
            "and add it to SimulationCfg.visualizer_cfgs.",
            DeprecationWarning,
            stacklevel=2,
        )

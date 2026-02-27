# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for marker utilities to simplify creation of UI elements in the GUI.

Currently, the sub-package provides the following classes:

* :class:`VisualizationMarkers` for creating a group of markers using `UsdGeom.PointInstancer
  <https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html>`_.


.. note::

    For some simple use-cases, it may be sufficient to use the debug drawing utilities from Isaac Sim.
    The debug drawing API is available in the `isaacsim.util.debug_drawing`_ module. It allows drawing of
    points and splines efficiently on the UI.

    .. _isaacsim.util.debug_drawing: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_debug_drawing.html

"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .visualization_markers import VisualizationMarkers
    from .visualization_markers_cfg import VisualizationMarkersCfg
    from .config import *  # noqa: F403

from isaaclab.utils.module import lazy_export

lazy_export(
    ("visualization_markers", ["VisualizationMarkers"]),
    ("visualization_markers_cfg", ["VisualizationMarkersCfg"]),
    submodules=["config"],
)

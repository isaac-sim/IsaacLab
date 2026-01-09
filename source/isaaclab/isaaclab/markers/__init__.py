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

# marker configurations
from .config import (
    BLUE_ARROW_X_MARKER_CFG,
    CONTACT_SENSOR_MARKER_CFG,
    CUBOID_MARKER_CFG,
    DEFORMABLE_TARGET_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    POSITION_GOAL_MARKER_CFG,
    RAY_CASTER_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
    SPHERE_MARKER_CFG,
    VISUO_TACTILE_SENSOR_MARKER_CFG,
)

# visualization markers
from .visualization_markers import VisualizationMarkers, VisualizationMarkersCfg

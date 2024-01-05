# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This submodule provides marker utilities for simplifying creation of UI elements in the GUI.

Currently, the module provides two classes:

* :class:`StaticMarker` for creating a group of markers from a single USD file.
* :class:`PointMarker` for creating a group of spheres.


.. note::

    For some simple usecases, it may be sufficient to use the debug drawing utilities from Isaac Sim.
    The debug drawing API is available in the `omni.isaac.debug_drawing`_ module. It allows drawing of
    points and splines efficiently on the UI.

    .. _omni.isaac.debug_drawing: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_debug_drawing.html

"""

from .point_marker import PointMarker
from .static_marker import StaticMarker

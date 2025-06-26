# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module provides utilities to create different terrains as height fields (HF).

Height fields are a 2.5D terrain representation that is used in robotics to obtain the
height of the terrain at a given point. This is useful for controls and planning algorithms.

Each terrain is represented as a 2D numpy array with discretized heights. The shape of the array
is (width, length), where width and length are the number of points along the x and y axis,
respectively. The height of the terrain at a given point is obtained by indexing the array with
the corresponding x and y coordinates.

.. caution::

    When working with height field terrains, it is important to remember that the terrain is generated
    from a discretized 3D representation. This means that the height of the terrain at a given point
    is only an approximation of the real height of the terrain at that point. The discretization
    error is proportional to the size of the discretization cells. Therefore, it is important to
    choose a discretization size that is small enough for the application. A larger discretization
    size will result in a faster simulation, but the terrain will be less accurate.

"""

from .hf_terrains_cfg import (
    HfDiscreteObstaclesTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfTerrainBaseCfg,
    HfWaveTerrainCfg,
)

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module provides methods to create different terrains using the ``trimesh`` library.

In contrast to the height-field representation, the trimesh representation does not
create arbitrarily small triangles. Instead, the terrain is represented as a single
tri-mesh primitive. Thus, this representation is more computationally and memory
efficient than the height-field representation, but it is not as flexible.
"""

from __future__ import annotations

from .mesh_terrains_cfg import (
    MeshBoxTerrainCfg,
    MeshFloatingRingTerrainCfg,
    MeshGapTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPitTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRailsTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshRepeatedCylindersTerrainCfg,
    MeshRepeatedPyramidsTerrainCfg,
    MeshStarTerrainCfg,
)

__all__ = [
    "MeshPlaneTerrainCfg",
    "MeshPyramidStairsTerrainCfg",
    "MeshInvertedPyramidStairsTerrainCfg",
    "MeshRandomGridTerrainCfg",
    "MeshRailsTerrainCfg",
    "MeshPitTerrainCfg",
    "MeshBoxTerrainCfg",
    "MeshGapTerrainCfg",
    "MeshFloatingRingTerrainCfg",
    "MeshStarTerrainCfg",
    "MeshRepeatedPyramidsTerrainCfg",
    "MeshRepeatedBoxesTerrainCfg",
    "MeshRepeatedCylindersTerrainCfg",
]

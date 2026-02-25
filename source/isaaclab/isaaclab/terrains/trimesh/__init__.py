# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "mesh_terrains_cfg": [
            "MeshBoxTerrainCfg",
            "MeshFloatingRingTerrainCfg",
            "MeshGapTerrainCfg",
            "MeshInvertedPyramidStairsTerrainCfg",
            "MeshPitTerrainCfg",
            "MeshPlaneTerrainCfg",
            "MeshPyramidStairsTerrainCfg",
            "MeshRailsTerrainCfg",
            "MeshRandomGridTerrainCfg",
            "MeshRepeatedBoxesTerrainCfg",
            "MeshRepeatedCylindersTerrainCfg",
            "MeshRepeatedPyramidsTerrainCfg",
            "MeshStarTerrainCfg",
        ],
    },
)

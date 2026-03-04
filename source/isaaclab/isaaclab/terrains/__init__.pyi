# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "FlatPatchSamplingCfg",
    "SubTerrainBaseCfg",
    "TerrainGenerator",
    "TerrainGeneratorCfg",
    "TerrainImporter",
    "TerrainImporterCfg",
    "color_meshes_by_height",
    "create_prim_from_mesh",
    "HfDiscreteObstaclesTerrainCfg",
    "HfInvertedPyramidSlopedTerrainCfg",
    "HfInvertedPyramidStairsTerrainCfg",
    "HfPyramidSlopedTerrainCfg",
    "HfPyramidStairsTerrainCfg",
    "HfRandomUniformTerrainCfg",
    "HfSteppingStonesTerrainCfg",
    "HfTerrainBaseCfg",
    "HfWaveTerrainCfg",
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
]

from .sub_terrain_cfg import FlatPatchSamplingCfg, SubTerrainBaseCfg
from .terrain_generator import TerrainGenerator
from .terrain_generator_cfg import TerrainGeneratorCfg
from .terrain_importer import TerrainImporter
from .terrain_importer_cfg import TerrainImporterCfg
from .utils import color_meshes_by_height, create_prim_from_mesh
from .height_field import (
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
from .trimesh import (
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

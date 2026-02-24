# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package with utilities for creating terrains procedurally.

There are two main components in this package:

* :class:`TerrainGenerator`: This class procedurally generates terrains based on the passed
  sub-terrain configuration. It creates a ``trimesh`` mesh object and contains the origins of
  each generated sub-terrain.
* :class:`TerrainImporter`: This class mainly deals with importing terrains from different
  possible sources and adding them to the simulator as a prim object.
  The following functions are available for importing terrains:

  * :meth:`TerrainImporter.import_ground_plane`: spawn a grid plane which is default in Isaac Sim.
  * :meth:`TerrainImporter.import_mesh`: spawn a prim from a ``trimesh`` object.
  * :meth:`TerrainImporter.import_usd`: spawn a prim as reference to input USD file.

"""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["height_field", "trimesh"],
    submod_attrs={
        "sub_terrain_cfg": ["FlatPatchSamplingCfg", "SubTerrainBaseCfg"],
        "terrain_generator": ["TerrainGenerator"],
        "terrain_generator_cfg": ["TerrainGeneratorCfg"],
        "terrain_importer": ["TerrainImporter"],
        "terrain_importer_cfg": ["TerrainImporterCfg"],
        "utils": ["color_meshes_by_height", "create_prim_from_mesh"],
        "trimesh": [
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
        "height_field": [
            "HfDiscreteObstaclesTerrainCfg",
            "HfInvertedPyramidSlopedTerrainCfg",
            "HfInvertedPyramidStairsTerrainCfg",
            "HfPyramidSlopedTerrainCfg",
            "HfPyramidStairsTerrainCfg",
            "HfRandomUniformTerrainCfg",
            "HfSteppingStonesTerrainCfg",
            "HfTerrainBaseCfg",
            "HfWaveTerrainCfg",
        ],
    },
)

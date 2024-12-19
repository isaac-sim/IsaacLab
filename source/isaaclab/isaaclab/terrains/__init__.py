# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package with utilities for creating terrains procedurally.

There are two main components in this package:

* :class:`TerrainGenerator`: This class procedurally generates terrains based on the passed
  sub-terrain configuration. It creates a ``trimesh`` mesh object and contains the origins of
  each generated sub-terrain.
* :class:`TerrainImporter`: This class mainly deals with importing terrains from different
  possible sources and adding them to the simulator as a prim object. It also stores the
  terrain mesh into a dictionary called :obj:`TerrainImporter.warp_meshes` that later can be used
  for ray-casting. The following functions are available for importing terrains:

  * :meth:`TerrainImporter.import_ground_plane`: spawn a grid plane which is default in isaacsim/isaaclab.
  * :meth:`TerrainImporter.import_mesh`: spawn a prim from a ``trimesh`` object.
  * :meth:`TerrainImporter.import_usd`: spawn a prim as reference to input USD file.

"""

from .height_field import *  # noqa: F401, F403
from .terrain_generator import TerrainGenerator
from .terrain_generator_cfg import FlatPatchSamplingCfg, SubTerrainBaseCfg, TerrainGeneratorCfg
from .terrain_importer import TerrainImporter
from .terrain_importer_cfg import TerrainImporterCfg
from .trimesh import *  # noqa: F401, F403
from .utils import color_meshes_by_height, create_prim_from_mesh

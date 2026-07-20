Added
^^^^^

* Added :attr:`~isaaclab.terrains.TerrainImporter.env_origins_wp`,
  :attr:`~isaaclab.terrains.TerrainImporter.terrain_levels_wp`, and
  :meth:`~isaaclab.terrains.TerrainImporter.update_env_origins_mask` for zero-copy Warp terrain
  access and mask-based curriculum updates.

Fixed
^^^^^

* Fixed identity quaternion initialization for uniform pose commands.

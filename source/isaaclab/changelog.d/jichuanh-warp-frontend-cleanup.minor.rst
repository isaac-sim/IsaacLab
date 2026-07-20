Added
^^^^^

* Added :attr:`~isaaclab.scene.InteractiveScene.env_origins_wp` for zero-copy Warp access to
  scene origins.
* Added :meth:`~isaaclab.terrains.TerrainImporter.update_env_origins_mask` for mask-based Warp
  terrain curriculum updates.

Changed
^^^^^^^

* Changed :attr:`~isaaclab.terrains.TerrainImporter.terrain_origins`,
  :attr:`~isaaclab.terrains.TerrainImporter.env_origins`,
  :attr:`~isaaclab.terrains.TerrainImporter.terrain_levels`, and
  :attr:`~isaaclab.terrains.TerrainImporter.terrain_types` to return
  :class:`~isaaclab.utils.warp.ProxyArray`. Use ``.torch`` for Torch operations
  and ``.warp`` for Warp kernels.

Fixed
^^^^^

* Fixed identity quaternion initialization for uniform pose commands.

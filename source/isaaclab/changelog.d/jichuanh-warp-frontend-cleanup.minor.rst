Added
^^^^^

* Added :attr:`~isaaclab.terrains.TerrainImporter.env_origins_wp`,
  :attr:`~isaaclab.terrains.TerrainImporter.terrain_levels_wp`, and
  :meth:`~isaaclab.terrains.TerrainImporter.update_env_origins_mask` for zero-copy Warp terrain
  access and mask-based curriculum updates.
* Added :attr:`~isaaclab.scene.InteractiveScene.env_origins_wp` and a boolean ``env_mask``
  option to :meth:`~isaaclab.scene.InteractiveScene.reset` for mask-based scene resets.

Changed
^^^^^^^

* Changed the uniform pose and velocity command debug visualization to shared private mixins
  reused by the Warp-native command terms.

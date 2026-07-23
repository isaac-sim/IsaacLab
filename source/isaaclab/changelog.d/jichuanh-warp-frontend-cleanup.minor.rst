Added
^^^^^

* Added :attr:`~isaaclab.terrains.TerrainImporter.env_origins_pa`,
  :attr:`~isaaclab.terrains.TerrainImporter.terrain_levels_pa`, and
  :meth:`~isaaclab.terrains.TerrainImporter.update_env_origins_mask` for zero-copy Warp terrain
  access and mask-based curriculum updates.
* Added :attr:`~isaaclab.scene.InteractiveScene.env_origins_pa` and a boolean ``env_mask``
  option to :meth:`~isaaclab.scene.InteractiveScene.reset` for mask-based scene resets.

Changed
^^^^^^^

* Changed the uniform pose and velocity command debug visualization to shared private mixins
  reused by the Warp-native command terms.
* Changed the :class:`~isaaclab.terrains.TerrainImporter` origin, level, and type buffers to
  :class:`~isaaclab.utils.warp.ProxyArray` storage: the Torch accessors and Warp views alias
  one owner, so consumer-held views stay pointer-stable across reconfiguration without
  per-buffer view caches.

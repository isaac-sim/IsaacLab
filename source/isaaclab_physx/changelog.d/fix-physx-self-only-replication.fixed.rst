* Fixed the PhysX replicator building a second, hidden body for a clone source that only its own environment
  uses, such as a :class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg` variant assigned to a single
  environment. The hidden body pushed the visible one out of its spawn pose. Layouts with such a source are now
  parsed from the stage.

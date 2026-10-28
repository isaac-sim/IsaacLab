* Fixed the PhysX replicator building a second, hidden body for a clone source that only its own environment
  uses, such as a :class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg` variant assigned to a single
  environment. The hidden body pushed the visible one out of its spawn pose. Such a source is now parsed from the
  stage, and the replicator still handles every other source.
* Fixed later scenes of a process reusing the PhysX stage-parse exclusions of the first scene that registered the
  replicator. After a scene with a USD-only asset, such as a spawned camera, a later scene with more environments
  got a second body for each object of the extra environments.

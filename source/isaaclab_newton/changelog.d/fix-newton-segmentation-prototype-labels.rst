Fixed
^^^^^

* Fixed the Newton Warp renderer reporting UNLABELLED semantic and instance segmentation for every
  environment but the prototype when the scene spawns only the prototype
  (:attr:`~isaaclab.sim.spawners.SpawnerCfg.spawn_path`) and relies on backend replication. Shapes
  cloned by the physics backend now resolve their labels through the prototype environment recorded
  in the clone plan, with the matched ancestor rebased into the cloned environment so instance ids
  stay distinct per environment.

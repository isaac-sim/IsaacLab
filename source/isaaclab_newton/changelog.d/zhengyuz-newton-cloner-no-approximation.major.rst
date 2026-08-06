Changed
^^^^^^^

* **Breaking:** Removed ``NewtonCfg.simplify_meshes``. Newton replication no longer
  approximates mesh colliders, so a USD-authored collision approximation survives
  cloning. Author the approximation on the asset instead, via
  :attr:`~isaaclab.sim.schemas.CollisionBaseCfg.mesh_collision_property` on the
  spawner config.

* Fixed :class:`~isaaclab_physx.physics.PhysxManager` never clearing its simulation views on invalidation, and the
  joint wrench sensor never reusing the articulation's view: views are keyed by prim path expression on each manager.

Added
^^^^^

* Added :class:`~isaaclab_physx.sim.schemas.PhysxCollisionCfg`, the ``physxCollision:*``
  single-namespace collision fragment (PhysX ``PhysxCollisionAPI``). It carries
  ``contact_offset`` / ``rest_offset`` plus the torsional patch-friction fields, and composes with
  :class:`~isaaclab.sim.schemas.UsdPhysicsCollisionCfg` in a ``collision_props`` fragment list.

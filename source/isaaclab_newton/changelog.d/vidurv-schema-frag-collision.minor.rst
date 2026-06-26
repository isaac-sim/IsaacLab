Added
^^^^^

* Added :class:`~isaaclab_newton.sim.schemas.NewtonCollisionCfg`, the ``newton:*``
  single-namespace collision fragment (``newton:contactMargin``, ``newton:contactGap`` via
  ``NewtonCollisionAPI``). It composes with
  :class:`~isaaclab.sim.schemas.UsdPhysicsCollisionCfg` and
  :class:`~isaaclab_physx.sim.schemas.PhysxCollisionCfg` in a ``collision_props`` fragment list.

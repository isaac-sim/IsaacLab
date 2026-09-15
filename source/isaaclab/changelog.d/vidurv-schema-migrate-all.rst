Changed
^^^^^^^

* Changed the core package's own physics-schema configuration call sites to the composable schema
  fragments (:class:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg`,
  :class:`~isaaclab.sim.schemas.UsdPhysicsCollisionCfg`, :class:`~isaaclab.sim.schemas.MassCfg`, and
  the ``isaaclab_physx`` counterparts) instead of the inheritance-based ``*PropertiesCfg`` classes.
  The legacy classes continue to work, so no configuration written against them needs to change.

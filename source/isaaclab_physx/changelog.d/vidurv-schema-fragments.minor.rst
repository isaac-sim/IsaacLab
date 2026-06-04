Added
^^^^^

* Added :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyCfg`, the ``physxRigidBody:*``
  single-namespace rigid-body fragment (PhysX ``PhysxRigidBodyAPI``). It carries the PhysX
  damping / velocity-limit / solver-iteration / sleep fields plus ``disable_gravity``.

Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_physx.sim.schemas.RigidBodyPropertiesCfg` from a
  deprecated cfg class into a deprecated factory that returns the equivalent fragment list
  ``[UsdPhysicsRigidBodyCfg(...), PhysxRigidBodyCfg(...)]``. Pass a fragment list to
  ``rigid_props`` instead of constructing ``RigidBodyPropertiesCfg``; code that mutated
  attributes on the returned object (e.g. ``cfg.spawn.rigid_props.disable_gravity = True``)
  must instead set the field on the ``PhysxRigidBodyCfg`` fragment in the list. The factory is
  scheduled for removal in 5.0.

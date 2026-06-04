Added
^^^^^

* Added :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyCfg`, the ``physxRigidBody:*``
  single-namespace rigid-body fragment (PhysX ``PhysxRigidBodyAPI``). It carries the PhysX
  damping / velocity-limit / solver-iteration / sleep fields plus ``disable_gravity``, and
  composes with :class:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg` in a ``rigid_props``
  fragment list.

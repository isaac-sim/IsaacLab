* Migrated bundled task configurations off the deprecated ``RigidBodyMaterialCfg`` alias to
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg` (configurations using
  PhysX-only friction/restitution combine modes) or
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` (solver-common properties
  only), so loading these tasks no longer emits a ``DeprecationWarning``. The authored USD material
  attributes are unchanged.

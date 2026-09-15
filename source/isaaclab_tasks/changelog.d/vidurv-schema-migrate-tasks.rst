Changed
^^^^^^^

* Changed the task configurations to author physics schemas with composable schema fragments
  (e.g. :class:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg` plus
  :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyCfg`) instead of the combined legacy
  property configs. Spawner slots now take a fragment list, and the non-USD ``fix_root_link``
  and ``ensure_drives_exist`` knobs moved from the legacy config onto the spawner. The authored
  USD attributes are unchanged. Task configurations that copy these snippets should replace
  ``RigidBodyPropertiesCfg`` / ``CollisionPropertiesCfg`` / ``MassPropertiesCfg`` /
  ``ArticulationRootPropertiesCfg`` / ``JointDrivePropertiesCfg`` with the fragment for the USD
  namespace that owns each field.

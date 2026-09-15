Deprecated
^^^^^^^^^^

* Deprecated the inheritance-based schema cfg classes in favor of the single-namespace schema
  fragments. Each class now raises a ``DeprecationWarning`` on instantiation and will be removed in
  5.0. Replace :class:`~isaaclab.sim.schemas.MassPropertiesCfg` with
  :class:`~isaaclab.sim.schemas.MassCfg`; :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg` with
  ``[UsdPhysicsRigidBodyCfg(...), PhysxRigidBodyCfg(...)]``;
  :class:`~isaaclab.sim.schemas.CollisionBaseCfg` with
  ``[UsdPhysicsCollisionCfg(...), PhysxCollisionCfg(...)]``;
  :class:`~isaaclab.sim.schemas.JointDriveBaseCfg` with
  ``[UsdPhysicsDriveCfg(...), PhysxJointCfg(...)]``;
  :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg` with
  :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg` or
  :class:`~isaaclab_newton.sim.schemas.NewtonArticulationCfg`; and
  :class:`~isaaclab.sim.schemas.MeshCollisionBaseCfg`,
  :class:`~isaaclab.sim.schemas.BoundingCubePropertiesCfg` and
  :class:`~isaaclab.sim.schemas.BoundingSpherePropertiesCfg` with
  :class:`~isaaclab.sim.schemas.UsdPhysicsMeshCollisionCfg`. Spawner slots accept fragments
  directly, so ``rigid_props=RigidBodyBaseCfg(...)`` becomes
  ``rigid_props=[UsdPhysicsRigidBodyCfg(...)]``. The non-USD ``fix_root_link`` and
  ``ensure_drives_exist`` flags moved to arguments of
  :func:`~isaaclab.sim.schemas.apply_articulation_root_properties` and
  :func:`~isaaclab.sim.schemas.apply_joint_drive_properties`. Deformable cfgs are unaffected.
* Deprecated the ``define_*`` and ``modify_*`` schema writers in favor of the fragment-based
  ``apply_*`` writers, which take a prim-path expression and a list of fragments. Each writer now
  raises a ``DeprecationWarning`` when called and will be removed in 5.0. Replace
  ``define_rigid_body_properties`` / ``modify_rigid_body_properties`` with
  :func:`~isaaclab.sim.schemas.apply_rigid_body_properties`, and likewise for the collision, mass,
  articulation-root, joint-drive, mesh-collision and tendon families. The deformable writers are
  unaffected.
* Reworded the deprecation notices on the previously deprecated ``*PropertiesCfg`` schema aliases
  to point at the new fragment replacements instead of the intermediate split classes, keeping
  their existing 5.0 removal target so the whole legacy schema cfg surface is documented to be
  removed in the same release as the classes it forwards to. The material and tendon aliases are
  unaffected.

Changed
^^^^^^^

* Cleaned up the schema-cfg base classes to no longer carry PhysX namespace metadata.
  :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg`,
  :class:`~isaaclab.sim.schemas.CollisionBaseCfg`,
  :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg`, and
  :class:`~isaaclab.sim.schemas.JointDriveBaseCfg` now declare ``_usd_namespace = None`` and
  ``_usd_applied_schema = None``. Per-field PhysX overrides for fields whose only USD path
  today is the ``physx*:*`` namespace (``disable_gravity``, ``contact_offset``,
  ``rest_offset``, ``articulation_enabled``, ``max_velocity``) are declared via a new
  ``_usd_field_exceptions`` mapping ``applied_schema -> (namespace, {cfg_field: usd_attr})``.
  When any listed field is non-None at write time, the writer applies that schema and writes
  the attribute under the exception namespace; otherwise the schema is not stamped onto the
  prim. PhysX subclasses (:class:`PhysxRigidBodyPropertiesCfg`,
  :class:`PhysxCollisionPropertiesCfg`, :class:`PhysxArticulationRootPropertiesCfg`,
  :class:`PhysxJointDrivePropertiesCfg`) now self-declare ``_usd_namespace`` and
  ``_usd_applied_schema`` for their own fields. Observable behavior on standard inputs is
  unchanged.
* Consolidated the per-writer schema-application loop in
  :mod:`isaaclab.sim.schemas` into a single shared helper ``_apply_namespaced_schemas``.
  ``modify_articulation_root_properties``, ``modify_rigid_body_properties``,
  ``modify_collision_properties``, ``modify_joint_drive_properties``,
  ``modify_mesh_collision_properties``, and ``spawn_rigid_body_material`` all delegate to the
  helper after writing their typed-API ``UsdPhysics`` fields. The canonical exception-table
  + main-namespace gating logic now lives in one place instead of being duplicated across
  six call sites.

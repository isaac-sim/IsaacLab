Added
^^^^^

* Added :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`, the solver-common
  base class for rigid-body physics materials. Carries the ``UsdPhysics.MaterialAPI`` standard
  fields (``static_friction``, ``dynamic_friction``, ``restitution``). The PhysX-specific
  compliant-contact and combine-mode fields moved to
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`.
* Added :class:`~isaaclab.sim.schemas.CollisionBaseCfg`, the solver-common base class for
  collision properties. Carries :attr:`collision_enabled` (``UsdPhysics.CollisionAPI``) plus
  :attr:`contact_offset` and :attr:`rest_offset` whose USD attributes are PhysX-namespaced
  but are consumed by Newton's importer via the PhysX bridge resolver
  (``import_usd.py:2104, 2111``).
* Added :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg`, the solver-common base class
  for articulation root properties (``fix_root_link``, ``articulation_enabled``).
* Added :class:`~isaaclab.sim.schemas.MeshCollisionBaseCfg`, the solver-common base class for
  mesh collision properties carrying ``mesh_approximation_name`` (writes
  ``physics:approximation`` via :class:`UsdPhysics.MeshCollisionAPI`). The class-level
  ``_usd_applied_schema`` metadata replaces the deprecated ``usd_api`` / ``physx_api``
  instance-field dispatch.

Changed
^^^^^^^

* Moved the ``max_velocity`` field from :class:`~isaaclab_physx.sim.schemas.PhysxJointDrivePropertiesCfg`
  to :class:`~isaaclab.sim.schemas.JointDriveBaseCfg`. The field is the only USD path to set
  Newton's ``Model.joint_velocity_limit`` and is consumed by Newton's importer. The USD
  attribute written is unchanged (``physxJoint:maxJointVelocity``); existing code using
  ``PhysxJointDrivePropertiesCfg(max_velocity=...)`` continues to work because the field
  is inherited.
* Moved the ``disable_gravity`` field from :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyPropertiesCfg`
  to :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg`. PhysX honors per-body via
  ``physxRigidBody:disableGravity``; Newton currently honors at scene level (partial),
  documented in the field docstring. Existing code using
  ``PhysxRigidBodyPropertiesCfg(disable_gravity=...)`` continues to work via inheritance.
* Documented :attr:`~isaaclab.sim.schemas.ArticulationRootPropertiesCfg.articulation_enabled`
  and :attr:`~isaaclab.sim.schemas.ArticulationRootPropertiesCfg.enabled_self_collisions`
  to lock their placement for the future :class:`ArticulationRootBaseCfg` /
  ``PhysxArticulationRootPropertiesCfg`` split: ``articulation_enabled`` stays on the base
  (single-namespace USD with verified Newton consumer); ``enabled_self_collisions`` moves
  to the PhysX subclass (dual-namespace USD, with a future Newton sibling cfg owning the
  ``newton:*`` namespace).
* Changed the defaults of :attr:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg.compliant_contact_stiffness`,
  :attr:`compliant_contact_damping`, :attr:`friction_combine_mode`, and
  :attr:`restitution_combine_mode` from concrete values (``0.0``, ``0.0``, ``"average"``,
  ``"average"``) to ``None``. PhysX engine defaults match the previous concrete values, so
  user-observable simulation behavior is unchanged; the difference is that these attributes
  are now authored on the prim only when the user explicitly sets them (consistent with the
  rest of the consumption-gated cfg layer).
* Relocated :class:`RigidBodyMaterialCfg` to :mod:`isaaclab_physx.sim.spawners.materials` and
  split its fields between the new :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`
  (UsdPhysics-standard friction/restitution) and
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`
  (PhysX-specific compliant-contact and combine-mode fields). A forwarding shim on
  :mod:`isaaclab.sim.spawners.materials` and :mod:`isaaclab.sim` preserves existing imports.
* Refactored :func:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material` to be
  metadata-driven: it reads ``_usd_applied_schema``, ``_usd_namespace``, and
  ``_usd_attr_name_map`` from the cfg class and gates ``PhysxMaterialAPI`` application on
  whether the user authored at least one PhysX-namespaced field with a non-``None`` value.
  Previously, the writer applied ``PhysxMaterialAPI`` unconditionally on every material spawn.
* Relocated :class:`CollisionPropertiesCfg` to :mod:`isaaclab_physx.sim.schemas` and split
  its fields between the new :class:`~isaaclab.sim.schemas.CollisionBaseCfg` (solver-common
  ``collision_enabled`` plus the PhysX-namespaced but Newton-consumed
  ``contact_offset`` / ``rest_offset``) and
  :class:`~isaaclab_physx.sim.schemas.PhysxCollisionPropertiesCfg` (PhysX-only
  ``torsional_patch_radius`` / ``min_torsional_patch_radius``). A forwarding shim on
  :mod:`isaaclab.sim.schemas`, :mod:`isaaclab.sim.schemas.schemas_cfg`, and
  :mod:`isaaclab.sim` preserves existing imports.
* Refactored :func:`~isaaclab.sim.schemas.modify_collision_properties` to be metadata-driven
  and to gate ``PhysxCollisionAPI`` application on whether the user authored at least one
  PhysX-namespaced field with a non-``None`` value. Previously, the writer applied
  ``PhysxCollisionAPI`` unconditionally on every collision prim, stamping the schema onto
  Newton-targeted prims that only set ``collision_enabled``.
* Relocated :class:`ArticulationRootPropertiesCfg` to :mod:`isaaclab_physx.sim.schemas` and
  split its fields between the new :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg`
  (solver-common ``fix_root_link`` plus the PhysX-namespaced ``articulation_enabled`` which
  is consumed by the IL Newton wrapper as a spawn-time guard) and
  :class:`~isaaclab_physx.sim.schemas.PhysxArticulationRootPropertiesCfg`
  (``enabled_self_collisions`` and PhysX TGS solver iter / sleep / stabilization thresholds).
  A forwarding shim on :mod:`isaaclab.sim.schemas`,
  :mod:`isaaclab.sim.schemas.schemas_cfg`, and :mod:`isaaclab.sim` preserves existing imports.
* Refactored :func:`~isaaclab.sim.schemas.modify_articulation_root_properties` to be
  metadata-driven and to gate ``PhysxArticulationAPI`` application on whether the user
  authored at least one PhysX-namespaced field with a non-``None`` value. Previously, the
  writer applied ``PhysxArticulationAPI`` unconditionally on every articulation root,
  stamping the schema onto Newton-targeted prims that only set ``fix_root_link``.
* Relocated :class:`MeshCollisionPropertiesCfg`, :class:`ConvexHullPropertiesCfg`,
  :class:`ConvexDecompositionPropertiesCfg`, :class:`TriangleMeshPropertiesCfg`,
  :class:`TriangleMeshSimplificationPropertiesCfg`, and :class:`SDFMeshPropertiesCfg` to
  :mod:`isaaclab_physx.sim.schemas`. :class:`BoundingCubePropertiesCfg` and
  :class:`BoundingSpherePropertiesCfg` stay in core because they author no PhysX schema.
  A forwarding shim preserves existing imports.
* Refactored :func:`~isaaclab.sim.schemas.modify_mesh_collision_properties` to be
  metadata-driven. The writer now reads ``_usd_applied_schema`` and ``_usd_namespace`` from
  the cfg class instead of consulting instance-level ``usd_api`` / ``physx_api`` fields.
  The standard :class:`UsdPhysics.MeshCollisionAPI` is always applied; PhysX cooking
  schemas (``PhysxConvexHullCollisionAPI`` etc.) are gated on at least one
  PhysX-namespaced tuning field being set.
* Relocated :class:`FixedTendonPropertiesCfg` and :class:`SpatialTendonPropertiesCfg` to
  :mod:`isaaclab_physx.sim.schemas` as :class:`PhysxFixedTendonPropertiesCfg` and
  :class:`PhysxSpatialTendonPropertiesCfg`. Tendons are a PhysX-only feature; no Newton
  equivalent exists. A forwarding shim on :mod:`isaaclab.sim.schemas`,
  :mod:`isaaclab.sim.schemas.schemas_cfg`, and :mod:`isaaclab.sim` preserves existing
  imports.

Deprecated
^^^^^^^^^^

* Deprecated the ``usd_api`` and ``physx_api`` instance attributes on the mesh-collision
  cfg classes in favor of class-level ``_usd_applied_schema`` metadata. Reading these
  attributes still works through one minor version but emits a ``DeprecationWarning``.
  Scheduled for removal in 5.0.

Fixed
^^^^^

* Fixed :meth:`~isaaclab.sim.schemas.modify_joint_drive_properties` and
  :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties` so that ``PhysxJointAPI`` and
  ``PhysxRigidBodyAPI`` are applied only when the user authored at least one PhysX-namespaced
  field with a non-``None`` value. Previously, schema application was gated on class-level
  metadata being defined, which caused Newton-targeted prims to receive PhysX schemas even
  when the user only set base ``UsdPhysics``-standard fields.

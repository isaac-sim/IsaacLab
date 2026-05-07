Added
^^^^^

* Added :class:`PhysxRigidBodyMaterialCfg`, a subclass of
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` carrying the
  ``PhysxMaterialAPI`` schema fields (``compliant_contact_stiffness``,
  ``compliant_contact_damping``, ``friction_combine_mode``, ``restitution_combine_mode``).
  Use this when authoring PhysX-specific material knobs; use the base class when only the
  UsdPhysics-standard friction/restitution fields are needed.
* Added :class:`PhysxCollisionPropertiesCfg`, a subclass of
  :class:`~isaaclab.sim.schemas.CollisionBaseCfg` carrying the PhysX-specific
  ``torsional_patch_radius`` / ``min_torsional_patch_radius`` friction approximations.
  These fields have no Newton equivalent.
* Added :class:`PhysxDeformableCollisionPropertiesCfg`, renaming the previous
  ``PhysXCollisionPropertiesCfg`` (capital X) for clarity. Used internally by
  :class:`DeformableBodyPropertiesCfg`.
* Added :class:`PhysxArticulationRootPropertiesCfg`, a subclass of
  :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg` carrying the PhysX-specific
  ``enabled_self_collisions``, ``solver_position_iteration_count``,
  ``solver_velocity_iteration_count``, ``sleep_threshold``, ``stabilization_threshold``.
* Added :class:`PhysxConvexHullPropertiesCfg`, :class:`PhysxConvexDecompositionPropertiesCfg`,
  :class:`PhysxTriangleMeshPropertiesCfg`,
  :class:`PhysxTriangleMeshSimplificationPropertiesCfg`, and
  :class:`PhysxSDFMeshPropertiesCfg` -- the PhysX-cooking-specific mesh collision
  subclasses. Each declares its own PhysxSchema cooking API via class-level
  ``_usd_applied_schema`` metadata and inherits ``mesh_approximation_name`` from
  :class:`~isaaclab.sim.schemas.MeshCollisionBaseCfg`.
* Added :class:`PhysxFixedTendonPropertiesCfg` and :class:`PhysxSpatialTendonPropertiesCfg`,
  the relocated PhysX-only tendon cfg classes. Same fields as the legacy core-side classes;
  no field-level split.

Changed
^^^^^^^

* Removed the ``max_velocity`` field and USD metadata
  (``_usd_applied_schema``, ``_usd_namespace``, ``_usd_attr_name_map``) from
  :class:`PhysxJointDrivePropertiesCfg`. The field moved to
  :class:`~isaaclab.sim.schemas.JointDriveBaseCfg`; ``PhysxJointDrivePropertiesCfg``
  inherits it. Existing instantiations continue to work unchanged.
* Removed the ``disable_gravity`` field from :class:`PhysxRigidBodyPropertiesCfg`.
  The field moved to :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg`;
  ``PhysxRigidBodyPropertiesCfg`` inherits it. Existing instantiations continue
  to work unchanged.

Deprecated
^^^^^^^^^^

* Deprecated :class:`RigidBodyMaterialCfg` in favor of
  :class:`PhysxRigidBodyMaterialCfg` (PhysX-specific) or
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` (solver-common).
  The legacy name remains as a concrete subclass of :class:`PhysxRigidBodyMaterialCfg`
  that emits ``DeprecationWarning`` on instantiation. Scheduled for removal in 5.0.
* Deprecated :class:`CollisionPropertiesCfg` in favor of
  :class:`PhysxCollisionPropertiesCfg` (PhysX-specific) or
  :class:`~isaaclab.sim.schemas.CollisionBaseCfg` (solver-common). The legacy name remains
  as a concrete subclass of :class:`PhysxCollisionPropertiesCfg` that emits
  ``DeprecationWarning`` on instantiation. Scheduled for removal in 5.0.
* Deprecated :class:`PhysXCollisionPropertiesCfg` (capital X, deformable-body) in favor of
  :class:`PhysxDeformableCollisionPropertiesCfg`. The capital-X name is preserved as a
  deprecation alias (concrete subclass) and is scheduled for removal in 5.0.
* Deprecated :class:`ArticulationRootPropertiesCfg` in favor of
  :class:`PhysxArticulationRootPropertiesCfg` (PhysX-specific) or
  :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg` (solver-common). The legacy name
  remains as a concrete subclass of :class:`PhysxArticulationRootPropertiesCfg` that emits
  ``DeprecationWarning`` on instantiation. Scheduled for removal in 5.0.
* Deprecated :class:`MeshCollisionPropertiesCfg`, :class:`ConvexHullPropertiesCfg`,
  :class:`ConvexDecompositionPropertiesCfg`, :class:`TriangleMeshPropertiesCfg`,
  :class:`TriangleMeshSimplificationPropertiesCfg`, and :class:`SDFMeshPropertiesCfg` in
  favor of :class:`~isaaclab.sim.schemas.MeshCollisionBaseCfg` or the new ``Physx*``
  subclasses. Legacy names remain as concrete subclasses that emit ``DeprecationWarning``
  on instantiation. Scheduled for removal in 5.0.
* Deprecated :class:`FixedTendonPropertiesCfg` in favor of
  :class:`PhysxFixedTendonPropertiesCfg`. Legacy name remains as a concrete subclass that
  emits ``DeprecationWarning`` on instantiation. Scheduled for removal in 5.0.
* Deprecated :class:`SpatialTendonPropertiesCfg` in favor of
  :class:`PhysxSpatialTendonPropertiesCfg`. Legacy name remains as a concrete subclass
  that emits ``DeprecationWarning`` on instantiation. Scheduled for removal in 5.0.

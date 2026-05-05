Added
^^^^^

* Added :class:`PhysxRigidBodyMaterialCfg`, a subclass of
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` carrying the
  ``PhysxMaterialAPI`` schema fields (``compliant_contact_stiffness``,
  ``compliant_contact_damping``, ``friction_combine_mode``, ``restitution_combine_mode``).
  Use this when authoring PhysX-specific material knobs; use the base class when only the
  UsdPhysics-standard friction/restitution fields are needed.

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

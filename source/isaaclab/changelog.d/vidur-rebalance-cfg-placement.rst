Added
^^^^^

* Added :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`, the solver-common
  base class for rigid-body physics materials. Carries the ``UsdPhysics.MaterialAPI`` standard
  fields (``static_friction``, ``dynamic_friction``, ``restitution``). The PhysX-specific
  compliant-contact and combine-mode fields moved to
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`.

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

Fixed
^^^^^

* Fixed :meth:`~isaaclab.sim.schemas.modify_joint_drive_properties` and
  :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties` so that ``PhysxJointAPI`` and
  ``PhysxRigidBodyAPI`` are applied only when the user authored at least one PhysX-namespaced
  field with a non-``None`` value. Previously, schema application was gated on class-level
  metadata being defined, which caused Newton-targeted prims to receive PhysX schemas even
  when the user only set base ``UsdPhysics``-standard fields.

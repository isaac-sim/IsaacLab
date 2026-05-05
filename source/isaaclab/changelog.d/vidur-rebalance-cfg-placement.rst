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

Fixed
^^^^^

* Fixed :meth:`~isaaclab.sim.schemas.modify_joint_drive_properties` and
  :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties` so that ``PhysxJointAPI`` and
  ``PhysxRigidBodyAPI`` are applied only when the user authored at least one PhysX-namespaced
  field with a non-``None`` value. Previously, schema application was gated on class-level
  metadata being defined, which caused Newton-targeted prims to receive PhysX schemas even
  when the user only set base ``UsdPhysics``-standard fields.

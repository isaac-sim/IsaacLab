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

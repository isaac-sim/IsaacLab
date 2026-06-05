Added
^^^^^

* Added the :class:`~isaaclab_physx.sim.schemas.PhysxJointCfg` joint-drive fragment
  (``physxJoint:*`` / ``PhysxJointAPI``), carrying ``max_joint_velocity`` (with the legacy
  ``max_velocity`` deprecation alias). Applied alongside
  :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` via
  :func:`~isaaclab.sim.schemas.apply_joint_drive_properties`.

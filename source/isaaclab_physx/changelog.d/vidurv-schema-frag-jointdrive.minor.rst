Added
^^^^^

* Added the :class:`~isaaclab_physx.sim.schemas.PhysxJointCfg` joint-drive fragment
  (``physxJoint:*`` / ``PhysxJointAPI``), carrying ``max_joint_velocity`` (with the legacy
  ``max_velocity`` deprecation alias). Applied alongside
  :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` via
  :func:`~isaaclab.sim.schemas.apply_joint_drive_properties`.
* Added :func:`~isaaclab_physx.sim.schemas.apply_physx_joint`, the dedicated applier for
  :class:`~isaaclab_physx.sim.schemas.PhysxJointCfg` that converts ``max_joint_velocity`` from
  rad/s to deg/s for angular (revolute) joints, matching the legacy joint-drive unit convention.

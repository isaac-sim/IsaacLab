* Fixed :class:`~isaaclab_newton.sim.schemas.MujocoJointCfg` with ``actuatorgravcomp=True``
  overwriting an explicitly authored body ``mjc:gravcomp`` of ``0.0`` with ``1.0``. Body-level
  gravity compensation is now auto-enabled only when the body has not authored ``mjc:gravcomp``,
  matching the deprecated :class:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg` path.

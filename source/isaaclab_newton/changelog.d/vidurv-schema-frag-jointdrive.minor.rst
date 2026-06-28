Added
^^^^^

* Added the :class:`~isaaclab_newton.sim.schemas.MujocoJointCfg` joint-drive fragment
  (``mjc:*`` / ``MjcJointAPI``), carrying joint-level ``actuatorgravcomp``. Applied alongside
  :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` via
  :func:`~isaaclab.sim.schemas.apply_joint_drive_properties`. The from-files spawn site continues
  to auto-enable body-level gravcomp for the fragment path.

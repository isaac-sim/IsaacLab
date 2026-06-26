Added
^^^^^

* Added :class:`~isaaclab_newton.sim.schemas.MujocoFixedTendonCfg` and its applier
  :func:`~isaaclab_newton.sim.schemas.apply_mujoco_fixed_tendon` for tuning ``mjc:*``
  fixed-tendon attributes on ``MjcTendon`` prims,
  splitting the Mujoco tune path out of the PhysX fixed-tendon applier.

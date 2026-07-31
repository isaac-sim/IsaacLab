Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.enable_rigid_soft_full_surface_contact`
  to generate edge and triangle-interior soft contacts against rigid SDFs, so rigid features that
  pass between soft vertices are caught.
* Added :class:`~isaaclab_newton.physics.NewtonShapeSDFCfg` and
  :attr:`~isaaclab_newton.physics.NewtonCfg.sdf_shape_cfgs` to provision volume SDFs on collider
  shapes selected by label regex, as required by full-surface rigid-soft contact.

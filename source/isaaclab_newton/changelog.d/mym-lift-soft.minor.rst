Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.enable_rigid_soft_full_surface_contact`
  to generate edge and triangle-interior soft contacts against full-surface-capable rigid colliders.
  Analytic shapes work directly; mesh and convex colliders require a volume SDF.

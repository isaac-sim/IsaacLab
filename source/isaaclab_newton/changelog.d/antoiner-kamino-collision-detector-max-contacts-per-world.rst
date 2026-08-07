Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.KaminoSolverCfg.collision_detector_max_contacts_per_world`
  to :class:`~isaaclab_newton.physics.KaminoSolverCfg`. When set, this overrides Newton's
  geometry-based contact capacity estimation for Kamino's internal collision detector and guarantees
  the collision pipeline is created regardless of whether the model has pre-computed explicit
  broadphase pairs.

Added
^^^^^

* Added :attr:`~isaaclab.physics.PhysicsCfg.deterministic`, a backend-agnostic request for
  reproducible physics. Each physics manager translates it into its own settings when the
  simulation starts and raises when its configuration cannot provide the guarantee. A
  backend-specific determinism attribute set explicitly takes precedence.

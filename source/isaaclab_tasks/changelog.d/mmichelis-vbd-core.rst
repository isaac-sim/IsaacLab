Changed
^^^^^^^

* Changed the Franka soft-body task configurations to use the core
  :class:`~isaaclab_newton.physics.VBDSolverCfg` and
  :attr:`~isaaclab_newton.physics.NewtonCfg.soft_contact_cfg`. Custom
  overrides should move nested ``solver_cfg.model_cfg`` values to the outer
  Newton configuration.

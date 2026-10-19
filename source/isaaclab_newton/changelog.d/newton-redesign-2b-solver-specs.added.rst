* Added solver specifications (:class:`~isaaclab_newton.physics.SolverSpec`, e.g.
  :data:`~isaaclab_newton.physics.MJWARP_SPEC`): functions and constants describing each Newton solver type to Isaac
  Lab. A solver configuration's :attr:`~isaaclab_newton.physics.NewtonSolverCfg.class_type` selects its spec, so
  ``instantiate(solver_cfg, model, deterministic_mode)`` builds Newton's solver, the only solver instance.

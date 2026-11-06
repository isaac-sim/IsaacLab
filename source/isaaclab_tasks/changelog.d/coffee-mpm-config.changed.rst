* Changed the coffee MPM solver to construct its native configuration without modifying the USD scene.
  Call :meth:`~isaaclab_newton.physics.MPMSolverCfg.author_usd` explicitly when exporting solver settings.

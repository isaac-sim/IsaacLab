* Deprecated the manual :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolver` and
  :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg` in favor of
  :class:`~isaaclab_newton.physics.coupling.CouplerProxyCfg` for new simulations. The manual solver retained its
  existing behavior during deprecation; switching to proxy coupling requires retuning because the force-exchange
  algorithms differ.
* Deprecated imports from :mod:`isaaclab_contrib.coupling` in favor of
  :mod:`isaaclab_newton.physics.coupling`, retaining the former package and submodule imports during migration.

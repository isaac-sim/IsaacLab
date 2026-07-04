Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.NewtonCfg.solver_reset` to clear
  solver-owned state after task-authored environment writes. The option is
  disabled by default and preserves authored positions and velocities.

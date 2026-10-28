* Changed the contrib Newton tasks with custom solvers (GB300 insertion, Rizon coffee) to
  :class:`~isaaclab_newton.physics.NewtonSolver` subclasses selected by their solver configurations, and the
  MPM tasks to :func:`~isaaclab_newton.physics.reset_mpm_solver_state`.

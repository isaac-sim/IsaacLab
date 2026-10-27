* Added :class:`~isaaclab_newton.physics.NewtonSolver`, Isaac Lab's generic Newton solver: it builds Newton's solver
  (:attr:`~isaaclab_newton.physics.NewtonSolver.native`) from a solver configuration and steps, resets and collides it
  through Newton's :class:`~newton.solvers.SolverBase` API. Each solver configuration's ``class_type`` selects a
  subclass, e.g. :class:`~isaaclab_newton.physics.MJWarpSolver`, that overrides only where its solver differs.

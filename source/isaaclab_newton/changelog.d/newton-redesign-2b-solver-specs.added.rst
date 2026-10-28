* Added :class:`~isaaclab_newton.physics.NewtonSolver`, whose stateless class methods build Newton's solver
  from a solver configuration and step, reset and collide it through Newton's :class:`~newton.solvers.SolverBase`
  API. Each solver configuration's ``class_type`` selects a subclass, e.g.
  :class:`~isaaclab_newton.physics.MJWarpSolver`, that overrides only where its solver differs.

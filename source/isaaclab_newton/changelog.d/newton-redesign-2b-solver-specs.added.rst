* Added Isaac Lab's generic Newton solver path: every solver is built, stepped, reset and collided through Newton's
  :class:`~newton.solvers.SolverBase` API, and its configuration (:class:`~isaaclab_newton.physics.NewtonSolverCfg`)
  names the Newton solver class (:attr:`~isaaclab_newton.physics.NewtonSolverCfg.solver_class`) and overrides
  capabilities and methods, such as ``build`` or ``reset``, only where the solver differs.

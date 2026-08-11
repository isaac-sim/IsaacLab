Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.physics.NewtonManager.reset_solver_state` under
  Implicit MPM rejecting Isaac Lab ``(world_count,)`` masks. The manager now
  expands them to the ``(world_count + 1,)`` shape required by
  :meth:`newton.solvers.SolverImplicitMPM.reset`, and skips selective masks when
  shared multi-world Implicit MPM cannot clear grid-backed warm starts per world.

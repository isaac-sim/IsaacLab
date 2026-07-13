Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_newton.physics.NewtonKaminoManager` to require exactly
  one articulation per environment. :meth:`~isaaclab_newton.physics.NewtonKaminoManager._build_solver`
  raises a ``RuntimeError`` at solver initialization when an environment contains multiple
  articulations. Multiple articulations per environment are not yet supported in IsaacLab's Kamino integration.

* Changed :class:`~isaaclab_newton.physics.NewtonManager` to route forward kinematics through a
  solver-specialized hook bound during solver initialization. Kamino overrides this hook to call
  :meth:`SolverKamino.reset` with :class:`SolverKamino.ResetConfig.from_joints` when
  :attr:`~isaaclab_newton.physics.KaminoSolverCfg.use_fk_solver` is enabled. Environment resets
  now share a single per-articulation mask for both :meth:`~isaaclab_newton.physics.NewtonManager.forward`
  and pre-step reconcile, replacing the separate per-world Kamino reset mask.

Fixed
^^^^^

* Fixed environment resets writing updated state into the wrong double-buffered simulation
  state when ``use_cuda_graph`` was disabled. With an odd number of substeps the canonical input
  state buffer flipped each step while asset write paths kept targeting the original binding, so
  reset environments stayed inconsistent for solvers with separate input/output states (e.g.
  :class:`~isaaclab_newton.physics.NewtonKaminoManager`).

* Fixed Kamino forward kinematics on environment resets leaving incorrect body poses for
  closed-loop systems. Reset environments are now always updated through Kamino's loop-closure FK
  solver instead of Newton's articulated ``eval_fk``.

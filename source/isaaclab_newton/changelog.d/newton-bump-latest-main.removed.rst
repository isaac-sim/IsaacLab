* **Breaking:** Removed ``VBDSolverCfg.rigid_compliant_alm``. Newton 1.9 always solves VBD rigid joints and
  contacts with compliant ALM, rejects ``False``, and ignores ``True``; omit the field.
* **Breaking:** Replaced the boolean ``NewtonSolver.deterministic`` class attribute with
  :attr:`~isaaclab_newton.physics.NewtonSolver.max_deterministic_mode`, the strongest Warp determinism mode the
  solver honors. Custom solvers that set ``deterministic = True`` must set ``max_deterministic_mode`` instead.

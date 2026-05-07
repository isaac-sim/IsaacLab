Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.physics.NewtonManager` to dispatch through
  solver-specific manager subclasses while preserving the existing
  ``NewtonCfg(solver_cfg=...)`` configuration pattern.

Deprecated
^^^^^^^^^^

* Deprecated :attr:`~isaaclab_newton.physics.NewtonSolverCfg.solver_type` for
  manager dispatch in favor of
  :attr:`~isaaclab_newton.physics.NewtonSolverCfg.class_type`. Existing configs
  remain valid, but new code should rely on ``class_type``.

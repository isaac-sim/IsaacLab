Changed
^^^^^^^

* **Breaking:** Replaced :class:`~isaaclab_newton.physics.KaminoSolverCfg` with
  :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` and
  :class:`~isaaclab_newton.physics.KaminoDVISolverCfg`. Select P-ADMM or DVI by
  constructing the matching config, and migrate solver settings to
  ``solver_cfg.dynamics_solver_cfg.<setting>``.

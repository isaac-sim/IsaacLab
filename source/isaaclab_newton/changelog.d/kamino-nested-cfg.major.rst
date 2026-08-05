Changed
^^^^^

* **Breaking:** Restructured :class:`~isaaclab_newton.physics.KaminoSolverCfg` into
  nested sub-configs (:class:`~isaaclab_newton.physics.KaminoPADMMCfg`,
  :class:`~isaaclab_newton.physics.KaminoDVICfg`,
  :class:`~isaaclab_newton.physics.KaminoDynamicsCfg`, and related nested classes)
  and added :attr:`~isaaclab_newton.physics.KaminoSolverCfg.dynamics_solver` to select
  PADMM or DVI. Migrate flat fields such as ``padmm_max_iterations`` to
  ``padmm.max_iterations``. For DVI, set ``dynamics_solver="dvi"`` with
  ``dynamics.preconditioning=False``. Use :func:`~isaaclab_newton.physics.kamino_padmm_solver_cfg`
  and :func:`~isaaclab_newton.physics.kamino_dvi_solver_cfg` for validated starting points.

Added
^^^^^

* Added :func:`~isaaclab_newton.physics.kamino_padmm_solver_cfg` and
  :func:`~isaaclab_newton.physics.kamino_dvi_solver_cfg` factory helpers for Kamino
  solver presets.

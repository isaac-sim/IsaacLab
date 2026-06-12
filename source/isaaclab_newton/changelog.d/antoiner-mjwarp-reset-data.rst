Fixed
^^^^^

* Fixed NaN values in MJWarp solver-internal buffers (``qacc_warmstart``,
  ``qfrc_applied``, ``xfrc_applied``, ``qacc``, contact arrays, energy, solver
  counters) persisting across env reset and re-diverging on the next solve.
  :class:`~isaaclab_newton.physics.NewtonMJWarpManager` now calls
  :func:`mujoco_warp.reset_data` with the accumulated per-world reset bitmask
  at the top of every :meth:`~isaaclab_newton.physics.NewtonManager.step`, so
  a world that produces a NaN can recover after
  :meth:`~isaaclab.envs.ManagerBasedEnv.reset`.  See
  https://github.com/newton-physics/newton/issues/1266 for the upstream
  discussion; this is the workaround until ``SolverMuJoCo.reset()`` lands in
  newton#2657.

Changed
^^^^^^^

* Added :meth:`~isaaclab_newton.physics.NewtonManager._reset_solver_internals`
  hook so each Newton solver subclass can clear its per-world internal scratch
  buffers at the top of :meth:`~isaaclab_newton.physics.NewtonManager.step`.
  Default is no-op; :class:`~isaaclab_newton.physics.NewtonMJWarpManager`
  overrides it to call :func:`mujoco_warp.reset_data`.
  :class:`~isaaclab_newton.physics.NewtonKaminoManager` continues to invoke
  its inline ``solver.reset(world_mask=...)`` from its own
  :meth:`~isaaclab_newton.physics.NewtonKaminoManager.step` override — a
  follow-up may migrate it to the same hook for uniformity.

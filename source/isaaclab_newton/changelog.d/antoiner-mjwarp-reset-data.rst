Fixed
^^^^^

* Fixed NaN values in MJWarp solver-internal buffers (``qacc_warmstart``,
  ``qfrc_applied``, ``xfrc_applied``, ``ctrl``, ``act``) persisting across
  env reset and re-diverging on the next solve.
  :class:`~isaaclab_newton.physics.NewtonMJWarpManager` now calls
  :meth:`SolverMuJoCo.reset` with the accumulated per-world reset mask
  whenever the reset masks are consumed (at the top of
  :meth:`~isaaclab_newton.physics.NewtonManager.step` and in
  :meth:`~isaaclab_newton.physics.NewtonManager.forward`), so a world that
  produces a NaN can recover after :meth:`~isaaclab.envs.ManagerBasedEnv.reset`.
  See https://github.com/newton-physics/newton/issues/1266 for the upstream
  discussion.

Changed
^^^^^^^

* Added :meth:`~isaaclab_newton.physics.NewtonManager._reset_solver_internals`
  hook that clears per-world solver-internal scratch buffers before the
  accumulated reset masks are consumed by
  :meth:`~isaaclab_newton.physics.NewtonManager.step` or
  :meth:`~isaaclab_newton.physics.NewtonManager.forward`. The default
  implementation forwards to :meth:`SolverBase.reset` with ``flags=0``,
  preserving the authored joint state — a no-op for solvers that do not
  implement ``reset()``, and automatic coverage for any solver that does.
  :class:`~isaaclab_newton.physics.NewtonMJWarpManager` specializes it to
  gate the non-mask-aware CPU-MuJoCo path;
  :class:`~isaaclab_newton.physics.NewtonKaminoManager` opts out because its
  forward-kinematics delegate already routes through
  :meth:`SolverKamino.reset`.

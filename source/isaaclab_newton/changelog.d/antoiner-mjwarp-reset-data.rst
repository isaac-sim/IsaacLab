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
* Changed the dtype of ``NewtonManager._world_reset_mask`` from ``wp.int32``
  to ``wp.bool`` so :func:`mujoco_warp.reset_data` (which strictly requires
  ``wp.array[bool]``) consumes it without conversion.
  :class:`~isaaclab_newton.physics.NewtonKaminoManager` maintains an int32
  mirror (``_world_reset_mask_int32``) and copies the bool source into it
  each step before calling ``solver.reset(world_mask=...)``, because the
  current upstream Kamino kernels are declared ``wp.array[int32]`` despite
  ``SolverKamino.reset``'s docstring advertising ``wp.int8 | wp.bool``.
  Tracked upstream at newton-physics/newton#2932; once Kamino widens its
  kernel signatures the int32 mirror can be removed and the bool mask passed
  through zero-copy.

* Added the functional core :mod:`isaaclab_newton.physics.newton_backend`. :class:`~isaaclab_newton.physics.NewtonBackend`
  now holds everything bound to one finalized model (state, control, solver, contacts, sensors, actuators, step
  callbacks, and the compiled :class:`~isaaclab_newton.physics.StepGraph`), and free functions such as
  ``init_solver``, ``invalidate_fk``, ``forward``, ``register_step_callback``, ``step``, and ``record_step`` take it
  explicitly. Several backends, with different solvers, can be built and stepped side by side.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_step_callback` and
  :meth:`~isaaclab_newton.physics.NewtonManager.unregister_step_callback` with
  :class:`~isaaclab_newton.physics.StepPhase` (``CONTROL``, ``POST_ACTUATOR``, ``STATE_FORCE``, ``POST_STEP``).
  Graphable callbacks are captured with the solver; others run eagerly at the same position.
* Added whole-environment-step capture: :meth:`~isaaclab_newton.physics.NewtonManager.step` called while the caller
  records a CUDA graph records the Newton step into that graph. Call
  :meth:`~isaaclab_newton.physics.NewtonManager.prepare` before the capture.
* Added an instance-owned control pipeline that records ordinary action terms, Torch actuators, native Newton
  actuators, external wrenches, solver steps, and native feedback updates in one graph when all operations support
  capture. Unsupported operations run eagerly in their declared order between captured segments.
* Added standalone ``NewtonManager(builder, cfg, dt=..., device=...)`` construction. Managers own independent
  callbacks, sites, builders, views, and backends. Custom managers can override ``finalize_backend`` and be selected
  through ``NewtonCfg.class_type`` or supplied to ``SimulationContext(physics_manager=manager)``.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.require_env_decimation` for consumers that need host work
  between physics steps, :meth:`~isaaclab_newton.physics.NewtonManager.view_row_worlds`, and a ``row_worlds``
  argument to :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_body_state`, so reset masks map view rows to
  worlds.
* Added ``newton_backend.invalidate_worlds`` to commit a world-masked reset from a device mask, so the reset can be
  recorded into a caller's CUDA graph and replayed with a different mask each time, and
  :attr:`~isaaclab_newton.physics.NewtonSolver.supports_heterogeneous_worlds` for solvers that step worlds with
  different contents. ``record_step`` needs no eager warm-up step and works inside Torch-owned captures that Warp
  joins with ``wp.capture_begin(stream, external=True)``.

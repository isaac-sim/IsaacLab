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
* Added a control-callback path to :class:`~isaaclab_newton.envs.mdp.actions.NewtonOperationalSpaceControllerAction`:
  on Newton physics the controller computes efforts inside the captured step before every physics step, so physics
  keeps running the whole decimation loop.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.require_env_decimation` for consumers that need host work
  between physics steps, :meth:`~isaaclab_newton.physics.NewtonManager.view_row_worlds`, and a ``row_worlds``
  argument to :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_body_state`, so reset masks map view rows to
  worlds.

* Added the functional core :mod:`isaaclab_newton.physics.newton_backend`: a
  :class:`~isaaclab_newton.physics.NewtonBackend` holds one finalized model and its solver, and free functions such as
  ``forward``, ``step`` and ``record_step`` take it explicitly, so several backends can step side by side.
* Added standalone ``NewtonManager(builder, cfg, dt=..., device=...)`` construction,
  :func:`~isaaclab_newton.physics.get_newton_builder` and :func:`~isaaclab_newton.physics.get_newton_backend`.
* Added step callbacks (``register_pre_step_callback``, ``register_control_callback``,
  ``register_post_actuator_callback``, ``register_state_force_callback``, ``register_post_step_callback``) captured
  with the solver when graphable, and :meth:`~isaaclab_newton.physics.NewtonManager.step_ops` for capturing the
  physics step together with the environment step.

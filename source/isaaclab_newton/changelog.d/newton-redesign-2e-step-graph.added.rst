* Added named step callbacks on :class:`~isaaclab_newton.physics.NewtonManager`
  (``register_pre_step_callback``, ``register_control_callback``, ``register_post_actuator_callback``,
  ``register_state_force_callback``, ``register_post_step_callback``), each returning a handle for
  ``unregister_step_callback``.

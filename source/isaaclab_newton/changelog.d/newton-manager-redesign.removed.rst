* **Breaking:** Removed ``NewtonManager.register_post_step_callback``, ``unregister_post_step_callback``,
  ``register_post_actuator_callback``, and ``register_state_force_callback``. Use
  :meth:`~isaaclab_newton.physics.NewtonManager.register_step_callback` with
  :attr:`~isaaclab_newton.physics.StepPhase.POST_STEP`, ``POST_ACTUATOR``, or ``STATE_FORCE``.
* **Breaking:** Removed ``NewtonManager.add_frame_transform_sensor``. The frame transformer constructs its
  :class:`newton.sensors.SensorFrameTransform` directly and samples it on read.
* Removed the unused ``NewtonManager.instantiate_builder_from_stage``.
* Removed the unused ``NewtonManager.get_num_envs``, ``get_dt``, ``get_solver_dt``, ``is_fabric_enabled``,
  ``provides_implicit_damping``, and ``add_world_builder_hook``. Use
  :func:`~isaaclab_newton.cloner.newton_builder_world_hook` to extend replicated worlds.

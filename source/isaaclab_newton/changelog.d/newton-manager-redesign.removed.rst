* **Breaking:** Removed ``NewtonManager.register_post_step_callback``, ``unregister_post_step_callback``,
  ``register_post_actuator_callback``, and ``register_state_force_callback``. Use
  :meth:`~isaaclab_newton.physics.NewtonManager.register_step_callback` with
  :attr:`~isaaclab_newton.physics.StepPhase.POST_STEP`, ``POST_ACTUATOR``, or ``STATE_FORCE``.
* **Breaking:** Removed ``NewtonManager.add_frame_transform_sensor``. The frame transformer constructs its
  :class:`newton.sensors.SensorFrameTransform` directly and samples it on read.
* Removed the unused ``NewtonManager.instantiate_builder_from_stage``.

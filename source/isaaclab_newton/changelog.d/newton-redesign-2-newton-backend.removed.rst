* **Breaking:** Removed ``NewtonManager.add_frame_transform_sensor``, ``unregister_post_step_callback``,
  ``get_num_envs``, ``get_dt``, ``get_solver_dt`` and ``NewtonQueries``. Construct
  :class:`newton.sensors.SensorFrameTransform` directly, unregister with the handle returned by
  ``register_post_step_callback``, and read ``backend.model.world_count``, ``backend.dt`` and ``backend.solver_dt``.

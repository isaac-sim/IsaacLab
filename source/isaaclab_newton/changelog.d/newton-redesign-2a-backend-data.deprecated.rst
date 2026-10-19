* Deprecated ``NewtonManager.get_model``, ``get_state_0``, ``get_state_1``, ``get_control``, ``get_contacts`` and
  ``get_solver``; read ``manager.backend.model`` and the other backend fields instead.
* Deprecated ``NewtonManager.get_num_envs``, ``get_dt`` and ``get_solver_dt``; read ``backend.model.world_count``,
  ``backend.dt`` and ``backend.solver_dt`` instead.
* Deprecated ``NewtonManager.cl_register_site`` in favor of
  :meth:`~isaaclab_newton.physics.NewtonManager.register_site`, and ``NewtonManager.add_frame_transform_sensor`` in
  favor of constructing :class:`newton.sensors.SensorFrameTransform` on ``manager.backend.model``.

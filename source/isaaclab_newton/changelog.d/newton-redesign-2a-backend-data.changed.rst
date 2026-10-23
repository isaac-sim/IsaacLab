* **Breaking:** Manager internals moved to backend fields: ``_solver``, ``_contacts``, ``_collision_pipeline``,
  ``_adapter``, ``_world_reset_mask``, ``_fk_reset_mask`` and ``kinematics_dirty`` are now ``backend.solver``,
  ``backend.contacts``, ``backend.collision_pipeline``, ``backend.actuators``, ``backend.world_mask``,
  ``backend.fk_mask`` and ``backend.kinematics_dirty``. Solver manager subclasses assign ``self.backend.solver`` in
  ``_build_solver``.
* **Breaking:** :meth:`~isaaclab_newton.physics.NewtonManager.add_contact_sensor` and
  :meth:`~isaaclab_newton.physics.NewtonManager.add_imu_sensor` return the Newton sensor instead of a key or index.
* **Breaking:** Site requests and extended-attribute requests persist on the shared builder across hard resets, so
  sensors no longer register them again. Replication outputs are read through the manager's accessors instead of
  private manager fields.
* **Breaking:** Removed ``NewtonManager.get_model``, ``get_state_0``, ``get_state_1``, ``get_control`` and
  ``get_contacts``; read ``manager.backend.model``, ``state_0``, ``state_1``, ``control`` and ``contacts``. Removed
  ``get_num_envs``, ``get_dt`` and ``get_solver_dt``; read ``backend.model.world_count``, ``backend.dt`` and
  ``backend.solver_dt``.
* **Breaking:** Removed ``NewtonManager.cl_register_site``; use
  :meth:`~isaaclab_newton.physics.NewtonManager.register_site`. Removed ``add_frame_transform_sensor``; construct
  :class:`newton.sensors.SensorFrameTransform` on ``manager.backend.model``.
* **Breaking:** :class:`~isaaclab_newton.physics.NewtonManager` takes its :class:`~isaaclab_newton.physics.NewtonCfg`
  at construction. :class:`~isaaclab_newton.physics.NewtonBackend` no longer takes a builder; it finalizes the
  simulation's shared builder. The ``_register_builder_attributes(builder, solver_cfg)`` subclass hook receives the
  solver configuration instead of reading the active simulation.
* **Breaking:** Moved :class:`~isaaclab_newton.physics.NewtonBackendCfg` to ``newton_backend_cfg`` and
  :class:`~isaaclab_newton.physics.NewtonBuilderCfg` to ``newton_builder_cfg``; import them from
  ``isaaclab_newton.physics``.

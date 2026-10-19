* **Breaking:** Under Newton physics, the physics manager owns its builder and backend instead of the simulation
  registry: ``sim.get_or_create_backend(NewtonBuilderCfg(...))`` raises, so use ``get_newton_builder(sim)`` and
  ``get_newton_backend(sim)``. :class:`~isaaclab_newton.physics.NewtonBackendCfg` takes the builder to finalize.
* **Breaking:** Manager internals moved to backend fields: ``_solver``, ``_contacts``, ``_collision_pipeline``,
  ``_adapter``, ``_world_reset_mask``, ``_fk_reset_mask`` and ``kinematics_dirty`` are now ``backend.solver``,
  ``backend.contacts``, ``backend.collision_pipeline``, ``backend.actuators``, ``backend.world_mask``,
  ``backend.fk_mask`` and ``backend.kinematics_dirty``. Solver manager subclasses assign ``self.backend.solver`` in
  ``_build_solver``.
* **Breaking:** :meth:`~isaaclab_newton.physics.NewtonManager.add_contact_sensor` and
  :meth:`~isaaclab_newton.physics.NewtonManager.add_imu_sensor` return the Newton sensor instead of a key or index.
* **Breaking:** Site requests and extended-attribute requests persist on the manager's builder across hard resets, so
  sensors no longer register them again. Replication outputs are read through
  :meth:`~isaaclab_newton.physics.NewtonManager.get_clone_record` instead of private manager fields, and the model no
  longer carries ``num_envs``; use ``model.world_count``.

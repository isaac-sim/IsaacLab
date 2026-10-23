* **Breaking:** Manager internals moved to backend fields: ``_solver``, ``_contacts``, ``_collision_pipeline``,
  ``_adapter``, ``_world_reset_mask``, ``_fk_reset_mask`` and ``kinematics_dirty`` are now ``backend.solver``,
  ``backend.contacts``, ``backend.collision_pipeline``, ``backend.actuators``, ``backend.world_mask``,
  ``backend.fk_mask`` and ``backend.kinematics_dirty``. Solver manager subclasses assign ``self.backend.solver`` in
  ``_build_solver``.
* **Breaking:** :meth:`~isaaclab_newton.physics.NewtonManager.add_contact_sensor` and
  :meth:`~isaaclab_newton.physics.NewtonManager.add_imu_sensor` return the Newton sensor instead of a key or index.
* **Breaking:** Site requests and extended-attribute requests persist on the shared builder across hard resets, so
  sensors no longer register them again. Replication outputs are read through
  :meth:`~isaaclab_newton.physics.NewtonManager.get_clone_record` instead of private manager fields.

* Added :class:`~isaaclab_newton.physics.NewtonBackend` fields for everything bound to one finalized Newton model under
  Newton physics: solver, collision pipeline, contacts, sensors, Newton actuators and reset masks. Read them from
  ``sim.backend_manager.backend``; a hard reset closes the backend and finalizes a new one.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_site` and the replication-output accessors
  :meth:`~isaaclab_newton.physics.NewtonManager.get_clone_record`,
  :meth:`~isaaclab_newton.physics.NewtonManager.get_site_index_map`,
  :meth:`~isaaclab_newton.physics.NewtonManager.get_world_xforms` and
  :meth:`~isaaclab_newton.physics.NewtonManager.get_clone_source_builders`
  (:class:`~isaaclab_newton.physics.NewtonCloneRecord`).

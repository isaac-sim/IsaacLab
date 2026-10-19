* Added :class:`~isaaclab_newton.physics.NewtonBackend` as the owner of everything bound to one finalized Newton model
  under Newton physics: state, control, solver, collision pipeline, contacts, sensors, Newton actuators and reset
  masks. Read them from ``sim.backend_manager.backend``; a hard reset closes the backend and finalizes a new one.
* Added standalone construction of Newton managers from a populated builder, e.g.
  ``NewtonMJWarpManager(builder, cfg, dt=0.01, device="cuda:0")``. Managers own their builder and backend and do not
  share state.
* Added :func:`~isaaclab_newton.physics.get_newton_builder` and :func:`~isaaclab_newton.physics.get_newton_backend`,
  which return the physics manager's builder and backend under Newton physics and the shared render-only ones
  otherwise.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_site` and the replication-output accessors
  :meth:`~isaaclab_newton.physics.NewtonManager.get_clone_record`,
  :meth:`~isaaclab_newton.physics.NewtonManager.get_site_index_map`,
  :meth:`~isaaclab_newton.physics.NewtonManager.get_world_xforms` and
  :meth:`~isaaclab_newton.physics.NewtonManager.get_clone_source_builders`
  (:class:`~isaaclab_newton.physics.NewtonCloneRecord`).

* **Breaking:** :class:`~isaaclab.physics.PhysicsManager` runtime methods and state are per instance.
  :class:`~isaaclab.sim.SimulationContext` constructs the manager selected by ``cfg.physics.class_type`` and owns it
  as ``sim.physics_manager``. Replace class-level calls such as ``PhysxManager.get_physics_sim_view()`` with
  ``sim.physics_manager.get_physics_sim_view()``; assets and sensors use ``self._physics_manager``.
* **Breaking:** Backend factories select implementations with
  :attr:`~isaaclab.physics.PhysicsManager.backend_name` instead of the manager class name, and
  :attr:`~isaaclab.physics.PhysicsManager.views` is keyed by prim path per manager instead of
  ``(manager class, path)``. Compare backends with ``sim.physics_manager.backend_name`` or :func:`isinstance`.

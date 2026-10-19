* **Breaking:** :class:`~isaaclab.physics.PhysicsManager` runtime methods and state are per instance.
  :class:`~isaaclab.sim.SimulationContext` constructs the manager selected by ``cfg.physics.class_type`` and owns it
  as :attr:`~isaaclab.sim.SimulationContext.backend_manager`. Call runtime methods on the instance, for example
  ``sim.backend_manager.get_physics_sim_view()`` instead of ``PhysxManager.get_physics_sim_view()``; assets and
  sensors use ``self._backend_manager``. Comparing ``sim.physics_manager`` with a manager class (``is``, ``==``,
  ``__name__``) no longer works; use :func:`isinstance` or :attr:`~isaaclab.physics.PhysicsManager.backend_name`.
* **Breaking:** Backend factories select implementations with
  :attr:`~isaaclab.physics.PhysicsManager.backend_name` instead of the manager class name, and
  :attr:`~isaaclab.physics.PhysicsManager.views` is keyed by prim path per manager instead of
  ``(manager class, path)``.
* **Breaking:** ``InteractiveScene.physics_backend`` is the backend key (``"physx"``,
  ``"ovphysx"``, ``"newton"``) instead of the lowercased manager class name (for example ``"newtonmjwarpmanager"``).

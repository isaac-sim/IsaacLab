* **Breaking:** Physics manager runtime methods are instance methods. Use ``sim.physics_manager`` to access the
  owner. ``SimulationContext`` accepts ``physics_manager=`` for an existing custom manager, while the physics
  configuration's ``class_type`` selects its default factory. Backend asset-data constructors receive their owner
  through the required ``physics_manager`` keyword.
* Added ``supports_graph_capture`` declarations to action terms and actuator models. Custom implementations opt in
  when all state changes are device operations and their execution can safely be replayed.
* :meth:`~isaaclab.physics.PhysicsManager.initialize` is no longer abstract: the base implementation attaches the
  context, so managers configured at construction need no override. A manager passed to
  :class:`~isaaclab.sim.SimulationContext` with its own physics configuration keeps it, and the context adopts it as
  ``cfg.physics``; previously the context's configuration silently replaced the manager's.

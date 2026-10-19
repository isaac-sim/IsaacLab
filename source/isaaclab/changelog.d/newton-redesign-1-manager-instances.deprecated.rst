* Deprecated class-level access to :class:`~isaaclab.physics.PhysicsManager` runtime methods and state, such as
  ``PhysxManager.get_physics_sim_view()``. It forwards to the active simulation's manager with a
  :class:`DeprecationWarning`.
* Deprecated :attr:`~isaaclab.sim.SimulationContext.physics_manager` in favor of
  :attr:`~isaaclab.sim.SimulationContext.backend_manager`.
* Deprecated the ``physics_manager`` argument of :meth:`~isaaclab.physics.PhysicsManager.safe_callback_invoke` in
  favor of ``backend_manager``.
* Deprecated passing a manager class or lazy class reference to
  :func:`~isaaclab.benchmark.stepping.profile_physics_steps`; pass the manager instance.

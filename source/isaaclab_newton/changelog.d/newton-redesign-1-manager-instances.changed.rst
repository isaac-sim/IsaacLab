* **Breaking:** :class:`~isaaclab_newton.physics.NewtonManager` and its solver managers are used as instances
  (``sim.physics_manager``); replace class-level calls such as ``NewtonManager.get_model()`` with
  ``sim.physics_manager.get_model()``. Newton asset and sensor data classes take the manager as a keyword-only
  ``physics_manager`` argument. Solver construction hooks (``_create_solver``, ``_filter_solver_kwargs``) take the
  determinism mode as an argument instead of reading manager state.

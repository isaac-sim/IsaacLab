* **Breaking:** :class:`~isaaclab_newton.physics.NewtonManager` and its solver managers are used as instances
  (``sim.backend_manager``); call runtime methods on the instance, for example ``sim.backend_manager.get_model()``
  instead of ``NewtonManager.get_model()``. Subclasses override runtime hooks as instance methods. Newton asset and
  sensor data classes take the manager as a keyword-only ``backend_manager`` argument. Solver construction hooks
  (``_create_solver``, ``_filter_solver_kwargs``) take the determinism mode as an argument instead of reading manager
  state.

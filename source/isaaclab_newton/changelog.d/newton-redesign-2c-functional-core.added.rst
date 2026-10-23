* Added the functions in :mod:`isaaclab_newton.physics.newton_backend` that take a
  :class:`~isaaclab_newton.physics.NewtonBackend` explicitly (``init_solver``, ``forward``, ``invalidate_fk``,
  ``mark_model_changed``, ``step``, ``run_query``, ``capture_graph``, ...), so independent backends with different
  solvers can be built and stepped side by side.

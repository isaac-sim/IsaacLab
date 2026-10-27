* Added the functions in :mod:`isaaclab_newton.physics.newton_backend` that take a
  :class:`~isaaclab_newton.physics.NewtonBackend` explicitly (``init_solver``, ``step``, ``forward``,
  ``invalidate_fk``, ``add_contact_sensor``, ``run_query``, ``capture_graph``, ...), so backends with different solvers
  can be built and stepped side by side.

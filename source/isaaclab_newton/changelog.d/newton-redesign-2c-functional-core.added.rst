* Added the functions in :mod:`isaaclab_newton.physics.newton_backend` that take a
  :class:`~isaaclab_newton.physics.NewtonBackend` as their first argument (``init_solver``, ``step``, ``forward``,
  ``invalidate_fk``, ``add_contact_sensor``, ``refit_bvh``, ``capture_graph``, ...), so backends with different
  solvers can be built and stepped side by side.
* Added an optional ``builder`` argument to :class:`~isaaclab_newton.physics.NewtonBackend` to finalize a builder
  other than the simulation's shared one.

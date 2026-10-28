* Added :class:`~isaaclab_newton.physics.NewtonBackend` methods that initialize, step and query one model
  (:meth:`~isaaclab_newton.physics.NewtonBackend.init_solver`, :meth:`~isaaclab_newton.physics.NewtonBackend.step`,
  :meth:`~isaaclab_newton.physics.NewtonBackend.forward`, :meth:`~isaaclab_newton.physics.NewtonBackend.invalidate_fk`,
  :meth:`~isaaclab_newton.physics.NewtonBackend.refit_bvh`, ...), so backends with different solvers can be built and
  stepped side by side.
* Added an optional ``builder`` argument to :class:`~isaaclab_newton.physics.NewtonBackend` to finalize a builder
  other than the simulation's shared one.

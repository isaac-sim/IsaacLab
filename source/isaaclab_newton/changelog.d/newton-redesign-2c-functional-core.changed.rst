* **Breaking:** Removed ``NewtonQueries``; use :func:`newton_backend.run_query
  <isaaclab_newton.physics.newton_backend.run_query>`, :func:`newton_backend.capture_graph
  <isaaclab_newton.physics.newton_backend.capture_graph>` and :func:`newton_backend.refit_bvh
  <isaaclab_newton.physics.newton_backend.refit_bvh>`.
* Changed :class:`~isaaclab_newton.physics.NewtonManager` to drive its backend through the ``newton_backend``
  functions. The backend now holds the step callbacks, pending model changes and the captured step, so a hard reset
  discards them with the model; consumers register callbacks again on ``PHYSICS_READY`` as before.
* Changed solver initialization to evaluate the initial forward kinematics before running the callbacks registered
  with :meth:`~isaaclab_newton.physics.NewtonManager.register_solver_init_callback`.

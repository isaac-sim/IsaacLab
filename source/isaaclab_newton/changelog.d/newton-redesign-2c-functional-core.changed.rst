* Changed :class:`~isaaclab_newton.physics.NewtonManager` to delegate to the ``newton_backend`` functions, so the
  backend holds the pre-actuator, post-actuator and post-step callbacks and the captured step.
  :meth:`~isaaclab_newton.physics.NewtonManager.reset` initializes in one pass: ``MODEL_INIT``, finalize,
  ``PHYSICS_READY``, then solver and contacts. :class:`~isaaclab_newton.physics.NewtonQueries` forwards to
  ``newton_backend.run_query`` and ``newton_backend.capture_graph``.

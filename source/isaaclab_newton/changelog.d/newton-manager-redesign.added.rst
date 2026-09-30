* Added :class:`~isaaclab_newton.physics.NewtonSolverBinding`, the single place for solver-specific construction,
  stepping, reset, and forward kinematics. Each Newton manager now only selects a binding.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.add_stage` and
  :meth:`~isaaclab_newton.physics.NewtonManager.remove_stage` with :class:`~isaaclab_newton.physics.StepPhase` to
  schedule consumer operations into the compiled step program. Graph-safe stages are captured with the solver;
  other stages run eagerly at the same position.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.get_schema`, an immutable
  :class:`~isaaclab_newton.physics.NewtonSchema` with world count, clone-plan world prototypes, and step timing.
* Added accessors for state consumers previously read from private attributes:
  :meth:`~isaaclab_newton.physics.NewtonManager.get_newton_backend`, ``get_solver``, ``get_actuator_adapter``,
  ``get_site_index_map``, ``get_world_xforms``, ``get_clone_source_builders``, ``build_requests``, and
  ``mark_particles_dirty``.
* Added a functional runtime API in :mod:`isaaclab_newton.physics.runtime`: :class:`~isaaclab_newton.physics.NewtonRuntime`
  is plain data bound to one Newton backend, and free functions such as ``create_runtime``, ``bind_solver``,
  ``invalidate_fk``, ``reconcile``, ``add_stage``, ``compile_program``, and ``step`` take it explicitly. Several
  runtimes on different backends can coexist; :class:`~isaaclab_newton.physics.NewtonManager` drives the active one.

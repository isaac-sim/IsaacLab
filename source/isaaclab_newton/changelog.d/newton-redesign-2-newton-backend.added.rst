* Added :class:`~isaaclab_newton.physics.NewtonBackend`, the data bound to one finalized Newton model (state, control,
  solver, contacts, sensors, actuators, step callbacks and the compiled step), and the functions in
  :mod:`isaaclab_newton.physics.newton_backend` that take it explicitly (``init_solver``, ``forward``,
  ``invalidate_fk``, ``mark_model_changed``, ``register_step_callback``, ``prepare``, ``step``, ``run_query``, ...),
  so independent backends with different solvers can be built and stepped side by side.
* Added stateless solver adapters (:class:`~isaaclab_newton.physics.NewtonSolver` and
  ``<Solver>SolverAdapter``), selected by ``solver_cfg.class_type``, and
  :func:`~isaaclab_newton.physics.get_newton_builder` / :func:`~isaaclab_newton.physics.get_newton_backend`.
* Added named step callbacks on :class:`~isaaclab_newton.physics.NewtonManager`
  (``register_pre_step_callback``, ``register_control_callback``, ``register_post_actuator_callback``,
  ``register_state_force_callback``, ``register_post_step_callback``), each returning a handle for
  ``unregister_step_callback``. One :class:`~isaaclab.utils.StepGraph` runs every :meth:`step`.
* Added standalone construction of :class:`~isaaclab_newton.physics.NewtonManager` from a populated builder.

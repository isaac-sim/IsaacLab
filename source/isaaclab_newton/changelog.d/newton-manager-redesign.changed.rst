* **Breaking:** Physics managers are instances; read the active one from ``sim.physics_manager``.
* **Breaking:** Solver hooks live in :class:`~isaaclab_newton.physics.NewtonSolver` adapter instances selected by
  ``solver_cfg.class_type``. Rename ``Newton<Solver>Manager`` to ``<Solver>SolverAdapter``, for example
  ``NewtonMJWarpManager`` to ``MJWarpSolverAdapter``, in the same ``*_manager`` modules.
* **Breaking:** :class:`~isaaclab_newton.physics.NewtonManager` owns its builder and backend. Use
  ``get_newton_builder(sim)`` and ``get_newton_backend(sim)`` instead of ``sim.get_or_create_backend(...)``.
* **Breaking:** :meth:`~isaaclab_newton.physics.NewtonManager.reset` builds the simulation in one pass;
  ``start_simulation`` and ``initialize_solver`` were removed. ``cl_register_site`` and
  ``activate_newton_actuator_path`` are now ``register_site`` and ``activate_actuators``, ``NewtonQueries`` is now
  ``newton_backend.run_query`` and ``capture_graph``, and the model's ``num_envs`` is ``Model.world_count``.
* **Breaking:** The full and physics-only stepping paths are one step graph; non-graphable Newton actuators run
  eagerly between captured segments.

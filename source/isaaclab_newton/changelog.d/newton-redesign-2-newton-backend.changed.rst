* **Breaking:** Solver-specific manager subclasses are replaced by stateless adapters in the same modules, for example
  ``NewtonMJWarpManager`` by ``MJWarpSolverAdapter``; ``NewtonCfg.class_type`` is ``NewtonManager`` independently of
  ``solver_cfg.class_type``. Custom solvers subclass :class:`~isaaclab_newton.physics.NewtonSolver` and implement
  ``create_solver(model, deterministic_mode)``.
* **Breaking:** Manager internals (``_solver``, ``_contacts``, ``_use_single_state``, ...) are replaced by backend
  fields, e.g. ``manager.backend.solver`` and ``manager.backend.solver_adapter.single_state``. Under Newton physics,
  get the builder and backend with ``get_newton_builder(sim)`` and ``get_newton_backend(sim)`` instead of
  ``sim.get_or_create_backend(NewtonBuilderCfg(...))``.
* **Breaking:** ``add_contact_sensor`` and ``add_imu_sensor`` return the Newton sensor; replication outputs are read
  through ``get_clone_record()``, ``get_site_index_map()``, ``get_world_xforms()`` and ``get_clone_source_builders()``.
  Use ``newton_backend.run_query`` and ``newton_backend.capture_graph`` instead of ``NewtonQueries``.
* :meth:`~isaaclab_newton.physics.NewtonManager.reset` initializes in one pass: ``MODEL_INIT``, finalize,
  ``PHYSICS_READY``, then solver and contacts.

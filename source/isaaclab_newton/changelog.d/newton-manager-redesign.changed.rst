* **Breaking:** Physics managers are instances. Read the active manager from ``sim.physics_manager`` instead of
  calling runtime methods on ``NewtonManager`` or a solver-manager class. Each manager owns construction and
  lifecycle state; ``NewtonBackend`` owns finalized simulation resources.
* **Breaking:** Solver hooks now live in ``NewtonSolver`` adapter instances, selected independently through
  ``solver_cfg.class_type`` and built with :func:`~isaaclab_newton.physics.create_solver_adapter`. An adapter holds its
  solver configuration as ``cfg`` and no runtime state; its hooks are instance methods, so builder hooks
  (``register_builder_attributes(builder)``, ``prepare_solver_builder(builder)``,
  ``get_usd_import_schema_resolvers()``) and ``create_solver(model, deterministic_mode)`` no longer take the solver
  configuration. ``NewtonBackend.solver_adapter`` and ``NewtonManager.solver_adapter`` are instances. Use
  :func:`~isaaclab_newton.physics.create_builder` instead of ``<Adapter>.create_builder``, and
  ``newton_solver.solver_kwargs`` instead of ``NewtonSolver.solver_kwargs``. For example, use ``MJWarpSolverAdapter`` in place of ``NewtonMJWarpManager``;
  ``NewtonCfg.class_type`` selects the integration manager. Apply the same ``*SolverAdapter`` rename for
  Featherstone, XPBD, VBD, Kamino, and MPM.
* **Breaking:** Under Newton physics, :class:`~isaaclab_newton.physics.NewtonManager` owns its builder
  (:attr:`~isaaclab_newton.physics.NewtonManager.builder`) and finalizes its backend directly instead of through the
  simulation registry. :class:`~isaaclab_newton.physics.NewtonBuilderCfg` and
  :class:`~isaaclab_newton.physics.NewtonBackendCfg` now describe only the render-only model used under other physics,
  and ``NewtonBackendCfg`` takes the ``builder`` to finalize. Use :func:`~isaaclab_newton.physics.get_newton_builder`
  and :func:`~isaaclab_newton.physics.get_newton_backend` instead of ``sim.get_or_create_backend(NewtonBuilderCfg(...))``
  and ``sim.get_or_create_backend(NewtonBackendCfg(...))``.
* **Breaking:** Replaced the separate full and physics-only stepping paths with one step graph. Newton actuators that
  are not CUDA-graph-safe run eagerly between captured segments instead of forcing the environment to own the
  decimation loop.
* **Breaking:** :meth:`~isaaclab_newton.physics.NewtonManager.reset` builds the whole simulation in one pass
  (``MODEL_INIT``, finalize, ``PHYSICS_READY``, solver), so ``start_simulation`` and ``initialize_solver`` were
  removed. :meth:`~isaaclab_newton.physics.NewtonManager.finalize_backend` applies site requests and solver-specific
  builder normalization before finalizing.
* **Breaking:** :meth:`~isaaclab_newton.physics.NewtonManager.add_contact_sensor` and
  :meth:`~isaaclab_newton.physics.NewtonManager.add_imu_sensor` return the Newton sensor instead of a key or index,
  and ``NewtonManager.transforms_may_change_on_graph_replay`` is now a method.
* Extended state and contact attribute requests apply directly to the shared builder.
* A determinism guarantee disables MuJoCo Warp's sensor stage instead of requiring
  ``MJWarpSolverCfg.disable_sensors``.
* **Breaking:** Renamed ``NewtonManager.cl_register_site`` to :meth:`~isaaclab_newton.physics.NewtonManager.register_site`
  and ``NewtonManager.activate_newton_actuator_path`` to
  :meth:`~isaaclab_newton.physics.NewtonManager.activate_actuators`, matching the functional core.
* **Breaking:** Replaced the ``NewtonQueries`` static-method class with the module functions ``run_query`` and
  ``capture_graph`` in :mod:`isaaclab_newton.physics.newton_backend`.
* **Breaking:** The Newton model no longer carries a patched ``num_envs`` attribute; use ``Model.world_count``.
* **Breaking:** ``MPMSolverAdapter.reset_solver_state`` is now the function
  :func:`~isaaclab_newton.physics.reset_mpm_solver_state`, and
  :meth:`~isaaclab_newton.physics.NewtonSolver.create_fixed_tendon_control` takes the model explicitly instead of
  reading the active one. ``newton_backend.refit_bvh`` is public. Solver hooks no longer read
  manager class state, so several backends can coexist.
* The Kamino manager resolves an automatic ``use_fk_solver`` into the backend's configuration instead of writing it
  into the user's configuration.
* The implicit MPM manager builds :class:`newton.solvers.SolverImplicitMPM.Config` directly from
  :class:`~isaaclab_newton.physics.MPMSolverCfg` instead of authoring ``NewtonMPMSceneAPI`` solver attributes on the
  physics scene and reading them back on every hard reset.
* The step count and capture function are resolved once per backend, in ``init_solver`` and
  :meth:`~isaaclab_newton.physics.NewtonManager.set_decimation`, so ``step(backend)`` reads
  ``NewtonBackend.steps_per_call`` and ``NewtonBackend.capture`` instead of re-deriving them every step.
  ``NewtonBackend.device`` is a :class:`warp.Device`.
* Post-actuator callbacks, such as actuator telemetry, run once per step, after the actuators of its last physics
  step.
* Forces authored before a step are staged and re-applied per substep only for double-buffered solvers or when
  state-force callbacks add forces; single-state solvers read them in place.
* Property edits queue model-change categories and a device-side dirty flag. ``forward`` or ``step`` commits the
  combined categories once. Empty reset masks skip constant reconstruction without synchronizing the device.
  Capture property-changing resets with ``newton_backend.capture_graph`` so conditional nodes retain their scratch
  storage. The helper returns a ``CapturedGraph`` with a ``launch()`` method.
* Post-step callbacks and native sensors update after every physics step, keeping feedback current within the
  captured decimation loop. Post-actuator telemetry remains sampled on the last physics step.
* A backend builds its own solver adapter from its configuration, and every adapter hook reads the solver settings
  from :attr:`~isaaclab_newton.physics.NewtonSolver.cfg`. Kamino keeps its resolved ``use_fk_solver`` on the adapter
  instead of rewriting the backend configuration. :attr:`~isaaclab_newton.physics.NewtonBackend.live` replaces the
  repeated ``is_stepping or device.is_capturing`` check in asset data.

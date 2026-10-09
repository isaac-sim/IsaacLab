* **Breaking:** Moved every model-bound object of :class:`~isaaclab_newton.physics.NewtonManager` (about 60 class
  attributes) onto :class:`~isaaclab_newton.physics.NewtonBackend`. The manager keeps only the active
  :attr:`~isaaclab_newton.physics.NewtonManager.backend`, site requests, replication outputs, and the decimation
  setting. Code that read private manager attributes must use the backend or the manager accessors.
* **Breaking:** Solver managers implement stateless classmethod hooks that take the backend explicitly
  (``create_solver``, ``step_solver``, ``reset_solver``, ``eval_fk``, ``uses_collision_pipeline``, and similar)
  instead of overriding private class hooks that wrote shared class state.
* **Breaking:** Replaced the separate full and physics-only stepping paths with one step graph. Newton actuators that
  are not CUDA-graph-safe run eagerly between captured segments instead of forcing the environment to own the
  decimation loop.
* **Breaking:** :meth:`~isaaclab_newton.physics.NewtonManager.reset` builds the whole simulation in one pass
  (``MODEL_INIT``, finalize, ``PHYSICS_READY``, solver), so ``start_simulation`` and ``initialize_solver`` were
  removed. :class:`~isaaclab_newton.physics.NewtonBackendCfg` now constructs the backend through
  ``create_newton_backend``, which applies site requests and solver-specific builder normalization before finalizing.
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
* **Breaking:** :meth:`~isaaclab_newton.physics.NewtonMPMManager.reset_solver_state` and
  :meth:`~isaaclab_newton.physics.NewtonManager.create_fixed_tendon_control` take the backend or model explicitly
  instead of reading the active one, and builder hooks receive the solver configuration. Solver hooks no longer read
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
* ``POST_ACTUATOR`` step callbacks, such as actuator telemetry, run once per step, after the actuators of its last
  physics step.
* Forces authored before a step are staged and re-applied per substep only for double-buffered solvers or when
  ``STATE_FORCE`` callbacks add forces; single-state solvers read them in place.
* Fixed-base root pose writes notify the solver through
  :meth:`~isaaclab_newton.physics.NewtonManager.add_model_change`. A model change authored while a caller records a
  CUDA graph notifies the solver immediately, so every replay applies it.

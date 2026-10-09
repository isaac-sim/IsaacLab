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

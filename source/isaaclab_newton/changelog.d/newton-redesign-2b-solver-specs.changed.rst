* **Breaking:** Removed the per-solver managers ``NewtonMJWarpManager``, ``NewtonXPBDManager``, ``NewtonVBDManager``,
  ``NewtonFeatherstoneManager``, ``NewtonKaminoManager`` and ``NewtonMPMManager``;
  :class:`~isaaclab_newton.physics.NewtonManager` runs every solver, and the solver configuration selects it.
  Custom solvers define a :class:`~isaaclab_newton.physics.SolverSpec` instead of overriding manager hooks, and solver
  capabilities are read from ``solver_spec(manager.backend.solver_cfg)``.
* **Breaking:** :attr:`~isaaclab_newton.physics.NewtonSolverCfg.class_type` now selects the solver specification
  instead of a manager class; set :attr:`~isaaclab_newton.physics.NewtonCfg.class_type` to select a manager.
* **Breaking:** Removed ``NewtonManager.create_builder``, ``create_fixed_tendon_control`` and
  ``NewtonMPMManager.reset_solver_state``; use :func:`~isaaclab_newton.physics.create_builder`,
  ``solver_spec(cfg).fixed_tendon_control(articulation, model)`` and
  :func:`~isaaclab_newton.physics.reset_mpm_solver_state`.
* **Breaking:** Renamed the solver modules after their contents, e.g. ``mjwarp_manager`` and ``mjwarp_manager_cfg`` to
  ``mjwarp_solver`` and ``mjwarp_solver_cfg``; import from ``isaaclab_newton.physics``.

* **Breaking:** Removed the per-solver managers ``NewtonMJWarpManager``, ``NewtonXPBDManager``, ``NewtonVBDManager``,
  ``NewtonFeatherstoneManager``, ``NewtonKaminoManager`` and ``NewtonMPMManager``;
  :class:`~isaaclab_newton.physics.NewtonManager` runs every solver. Custom solvers subclass the matching
  :class:`~isaaclab_newton.physics.NewtonSolverFunctions`, e.g.
  :class:`~isaaclab_newton.physics.MJWarpSolverFunctions`, instead of overriding manager hooks.
* **Breaking:** :attr:`~isaaclab_newton.physics.NewtonSolverCfg.class_type` now selects the
  :class:`~isaaclab_newton.physics.NewtonSolverFunctions` instead of a manager class; set
  :attr:`~isaaclab_newton.physics.NewtonCfg.class_type` to select a manager.
* **Breaking:** Removed ``NewtonManager.create_builder``, ``create_fixed_tendon_control`` and
  ``NewtonMPMManager.reset_solver_state``; use :func:`~isaaclab_newton.physics.create_builder`,
  :meth:`NewtonSolverFunctions.fixed_tendon_control
  <isaaclab_newton.physics.NewtonSolverFunctions.fixed_tendon_control>` and
  :func:`~isaaclab_newton.physics.reset_mpm_solver_state`.
* **Breaking:** Renamed the ``*_manager`` and ``*_manager_cfg`` solver modules to ``*_solver`` and ``*_solver_cfg``,
  e.g. ``mjwarp_manager`` to ``mjwarp_solver``; import from ``isaaclab_newton.physics``.
* Changed VBD to register its Newton custom builder attributes, like the other solvers.
* Changed the check that rejects :attr:`~isaaclab_newton.physics.NewtonCfg.collision_cfg` with MuJoCo contacts to
  apply to any solver that does not use Newton's collision pipeline, e.g. Kamino's own collision detector and MPM.
* Changed :attr:`~isaaclab_newton.physics.NewtonCfg.class_type` to reject ``None``; omit it to use
  :class:`~isaaclab_newton.physics.NewtonManager`.

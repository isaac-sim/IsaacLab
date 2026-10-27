* **Breaking:** Removed the per-solver managers ``NewtonMJWarpManager``, ``NewtonXPBDManager``, ``NewtonVBDManager``,
  ``NewtonFeatherstoneManager``, ``NewtonKaminoManager`` and ``NewtonMPMManager``;
  :class:`~isaaclab_newton.physics.NewtonManager` runs every solver. Custom solvers subclass their solver
  configuration and override its methods instead of manager hooks, and solver capabilities are read from
  ``manager.backend.solver_cfg``.
* **Breaking:** Removed ``NewtonSolverCfg.class_type``; set
  :attr:`~isaaclab_newton.physics.NewtonSolverCfg.solver_class` to change the Newton solver class, and
  :attr:`~isaaclab_newton.physics.NewtonCfg.class_type` to select a manager.
* **Breaking:** Removed ``NewtonManager.create_builder``, ``create_fixed_tendon_control`` and
  ``NewtonMPMManager.reset_solver_state``; use :func:`~isaaclab_newton.physics.create_builder`,
  ``solver_cfg.fixed_tendon_control(articulation, model)`` and
  :func:`~isaaclab_newton.physics.reset_mpm_solver_state`.
* **Breaking:** Removed the ``mjwarp_manager``, ``xpbd_manager``, ``vbd_manager``, ``featherstone_manager`` and
  ``kamino_manager`` modules, and renamed ``mpm_manager`` to ``mpm_solver`` and the ``*_manager_cfg`` modules to
  ``*_solver_cfg``; import from ``isaaclab_newton.physics``.
* Changed VBD to register its Newton custom builder attributes, like the other solvers.
* Changed the check that rejects :attr:`~isaaclab_newton.physics.NewtonCfg.collision_cfg` with MuJoCo contacts to
  apply to any solver that reports its own contacts.

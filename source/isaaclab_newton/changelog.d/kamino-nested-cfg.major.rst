Changed
^^^^^^^

* **Breaking:** Replaced :class:`~isaaclab_newton.physics.KaminoSolverCfg` with
  :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` and
  :class:`~isaaclab_newton.physics.KaminoDVISolverCfg`. Select P-ADMM or DVI by
  constructing the matching config, and migrate solver settings to
  ``solver_cfg.dynamics_solver_cfg.<setting>``.
* Changed the default Kamino P-ADMM configuration to use Moreau integration,
  automatic sparse-Jacobian selection, ``constraints.alpha=0.1``,
  ``max_iterations=100``, tolerances of ``1e-4``, ``rho_0=0.05``,
  ``contact_warmstart_method="geom_pair_net_force"``, and
  ``use_graph_conditionals=False``. Set the corresponding nested configuration
  fields explicitly to retain the previous behavior.

* **Breaking:** Replaced ``NewtonCouplerManager`` and ``NewtonCoupledMJWarpVBDManager`` with the solvers
  :class:`~isaaclab_contrib.coupling.CouplerSolver` and
  :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolver`, selected by the configurations'
  ``class_type``; each entry's solver is built from its own configuration. Tasks with a custom coupled solver subclass
  :class:`~isaaclab_contrib.coupling.CouplerSolver` and override ``build_coupled``.
* **Breaking:** Renamed ``custom_coupling.coupled_mjwarp_vbd_manager`` and ``custom_coupling.newton_manager_cfg`` to
  ``coupled_mjwarp_vbd_solver`` and ``coupled_mjwarp_vbd_solver_cfg``.
* Added :attr:`~isaaclab_contrib.coupling.CouplerCfg.outer_contact_entry` to collide only one entry's shape pairs in
  the outer collision pipeline. A :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` without proxies now steps its
  entries independently. ``{ENV_REGEX_NS}`` selectors expand against the environment template the manager passes to
  the coupler.
* Changed the coupler to forward :attr:`~isaaclab_newton.physics.NewtonCfg.debug_mode` diagnostics to each entry's
  solver.

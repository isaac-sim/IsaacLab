* **Breaking:** Removed ``NewtonCouplerManager`` and ``NewtonCoupledMJWarpVBDManager``;
  :class:`~isaaclab_contrib.coupling.CouplerCfg` and
  :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg` build and drive their solvers, recursing into
  each entry's own solver configuration. Tasks with a custom coupled solver override
  :meth:`~isaaclab_contrib.coupling.CouplerCfg.construct_solver`.
* **Breaking:** Renamed ``custom_coupling.coupled_mjwarp_vbd_manager`` and ``custom_coupling.newton_manager_cfg`` to
  ``coupled_mjwarp_vbd_solver`` and ``coupled_mjwarp_vbd_solver_cfg``.
* Added :attr:`~isaaclab_contrib.coupling.CouplerCfg.outer_contact_entry` to collide only one entry's shape pairs in
  the outer collision pipeline. A :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` without proxies now steps its
  entries independently. ``{ENV_REGEX_NS}`` selectors expand against the environment template the manager passes to
  the coupler.
* Changed :class:`~isaaclab_contrib.coupling.CouplerCfg` to forward :attr:`~isaaclab_newton.physics.NewtonCfg.debug_mode`
  diagnostics to each entry's solver.

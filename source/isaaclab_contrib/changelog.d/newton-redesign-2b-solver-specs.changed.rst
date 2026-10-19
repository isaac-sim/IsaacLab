* **Breaking:** Replaced ``NewtonCouplerManager`` and ``NewtonCoupledMJWarpVBDManager`` with the solver
  specifications :data:`~isaaclab_contrib.coupling.COUPLER_SPEC` and ``COUPLED_MJWARP_VBD_SPEC``, selected by the
  solver configuration's ``class_type``. The coupler builds and queries each entry through the entry's own spec.
* **Breaking:** Renamed ``custom_coupling.coupled_mjwarp_vbd_manager`` and ``custom_coupling.newton_manager_cfg`` to
  ``coupled_mjwarp_vbd_solver`` and ``coupled_mjwarp_vbd_solver_cfg``.
* Added :attr:`~isaaclab_contrib.coupling.CouplerCfg.outer_contact_entry` to collide only one entry's shape pairs in
  the outer collision pipeline, and :func:`~isaaclab_contrib.coupling.coupler_builder` for custom coupled solvers. A
  :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` without proxies now steps its entries independently.

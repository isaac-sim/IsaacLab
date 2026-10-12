* **Breaking:** Renamed ``NewtonCouplerManager`` and ``NewtonCoupledMJWarpVBDManager`` to
  :class:`~isaaclab_contrib.coupling.CouplerSolverAdapter` and ``CoupledMJWarpVBDSolverAdapter``, Newton solver
  adapters selected through the solver configuration. Import the latter from ``coupled_mjwarp_vbd_manager``.
* Adapted terms to mask-based environment resets.

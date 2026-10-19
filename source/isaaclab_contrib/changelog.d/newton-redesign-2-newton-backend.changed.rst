* **Breaking:** Replaced ``NewtonCouplerManager`` and ``NewtonCoupledMJWarpVBDManager`` with the stateless solver
  adapters :class:`~isaaclab_contrib.coupling.CouplerSolverAdapter` and ``CoupledMJWarpVBDSolverAdapter``, selected
  by the solver configuration's ``class_type``.

* **Breaking:** Changed :class:`~isaaclab_contrib.coupling.CouplerSolver` and
  :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolver` into solver drivers. The coupler kept each entry's
  driver in ``entries`` and ``build_coupled`` became an instance method; the custom coupled solver kept its sub-solver
  drivers as ``rigid`` and ``soft``.

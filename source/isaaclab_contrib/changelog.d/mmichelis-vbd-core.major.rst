Changed
^^^^^^^

* **Breaking:** Moved the standalone VBD solver from
  ``isaaclab_contrib.deformable`` to :mod:`isaaclab_newton.physics`. Import
  :class:`~isaaclab_newton.physics.NewtonVBDManager` and
  :class:`~isaaclab_newton.physics.VBDSolverCfg` from their new location, and
  move ``NewtonModelCfg`` and ``NewtonModelSolverCfg`` soft-contact settings to
  :attr:`~isaaclab_newton.physics.NewtonCfg.soft_contact_cfg`.
* **Breaking:** Removed the
  ``isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg`` compatibility
  alias. Import
  :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg` instead.

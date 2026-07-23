Added
^^^^^

* Added the opt-in :mod:`isaaclab_contrib.custom_coupling` example. Import the
  module explicitly to register ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling``.

Deprecated
^^^^^^^^^^

* Deprecated ``CoupledMJWarpVBDSolverCfg`` and the
  ``deformable.coupled_mjwarp_vbd_manager`` compatibility path. Use
  :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` for MJWarp and VBD
  coupling, or import :mod:`isaaclab_contrib.custom_coupling` for the opt-in
  manual MJWarp and VBD example.

Removed
^^^^^^^

* **Breaking:** Removed ``CoupledFeatherstoneVBDSolverCfg`` and
  ``NewtonCoupledFeatherstoneVBDManager`` from
  :mod:`isaaclab_contrib.deformable`. Switch the rigid solver to MJWarp and use
  :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` or the opt-in
  :mod:`isaaclab_contrib.custom_coupling` example.

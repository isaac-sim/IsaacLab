Added
^^^^^

* Added the opt-in :mod:`isaaclab_contrib.custom_coupling` example. Import the
  module explicitly to register ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling``.

Removed
^^^^^^^

* **Breaking:** Removed ``CoupledMJWarpVBDSolverCfg`` and
  ``CoupledFeatherstoneVBDSolverCfg``, and their managers, from
  :mod:`isaaclab_contrib.deformable`. Use
  :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` for MJWarp and VBD
  coupling, or import :mod:`isaaclab_contrib.custom_coupling` for the opt-in
  manual MJWarp and VBD example. Featherstone users must switch their rigid
  solver to MJWarp before migrating.

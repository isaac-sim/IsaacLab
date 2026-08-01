Added
^^^^^

* Added the opt-in :mod:`isaaclab_contrib.custom_coupling` example. Import
  :mod:`isaaclab_contrib.custom_coupling.tasks` explicitly to register
  ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling``.

Deprecated
^^^^^^^^^^

* Deprecated :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg`. Use
  :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` for MJWarp and VBD
  coupling, or :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg`
  to stay on the manual coupler.

Removed
^^^^^^^

* **Breaking:** Moved ``NewtonCoupledMJWarpVBDManager`` and its reaction kernel out
  of :mod:`isaaclab_contrib.deformable` and into the opt-in
  :mod:`isaaclab_contrib.custom_coupling` example, and removed the
  ``isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager`` module. Import the
  manager from :mod:`isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager`
  instead. Configurations that reference the manager through
  :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` keep working and
  now resolve to the moved class.

* **Breaking:** Removed ``CoupledFeatherstoneVBDSolverCfg`` and
  ``NewtonCoupledFeatherstoneVBDManager`` from
  :mod:`isaaclab_contrib.deformable`. Switch the rigid solver to MJWarp and use
  :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` or the opt-in
  :mod:`isaaclab_contrib.custom_coupling` example.

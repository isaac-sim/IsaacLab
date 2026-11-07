* Deprecated :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg` and
  :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolver` in favor of
  :class:`~isaaclab_newton.physics.coupling.CouplerProxyCfg`. The compatibility configuration now uses staggered
  proxy feedback instead of shared-model penalty reactions; retune existing contact and solver settings.
* Deprecated ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling`` and its ``newton_mjwarp_vbd`` preset. They now use
  the core proxy preset; use ``Isaac-Lift-Soft-Franka`` with ``physics=newton_mjwarp_vbd_proxy`` instead.
* Deprecated imports from :mod:`isaaclab_contrib.coupling` in favor of
  :mod:`isaaclab_newton.physics.coupling`, retaining the former package and submodule imports during migration.

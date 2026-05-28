Added
^^^^^

* Added :mod:`isaaclab_contrib.coupling`, exposing
  :class:`~isaaclab_contrib.coupling.coupled_manager.NewtonCoupledSolverManager`
  together with the
  :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledSolverCfg`
  base config and two algorithm-specific subclasses:
  :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledProxySolverCfg`
  (wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledProxy`) and
  :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledAdmmSolverCfg`
  (wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledAdmm`).
  The manager partitions the Newton model into source/destination entries,
  instantiates the sub-solvers from their cfg types, and dispatches on the
  config subclass to build the matching coupled solver.

* Added support for raw prim-path regex strings (e.g.
  ``"/World/envs/env_.*/MyCube"``) in the body-selector lists of
  :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledSolverCfg`,
  alongside :class:`~isaaclab.managers.SceneEntityCfg` entries.

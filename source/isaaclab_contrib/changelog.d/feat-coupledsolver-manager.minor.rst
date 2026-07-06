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
  (wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledADMM`).
  The manager partitions the Newton model among explicit, named
  :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledSolverEntryCfg`
  entries, instantiates each sub-solver from its config, and connects entries
  through named proxy mappings or ADMM contact pairs.

* Added support for raw prim-path regex strings (e.g.
  ``"/World/envs/env_.*/MyCube"``) in the body-selector lists of
  :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledSolverEntryCfg`
  and :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledProxyCfg`,
  alongside :class:`~isaaclab.managers.SceneEntityCfg` selectors.

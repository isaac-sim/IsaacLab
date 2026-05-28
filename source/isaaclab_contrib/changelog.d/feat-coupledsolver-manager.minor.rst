Added
^^^^^

* Added :mod:`isaaclab_contrib.coupling`, a new submodule housing
  :class:`~isaaclab_contrib.coupling.coupled_manager.NewtonCoupledSolverManager`
  along with the :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledSolverCfg`
  base config and the algorithm-specific
  :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledProxySolverCfg`
  (lagged-impulse virtual-proxy coupling, wrapping
  :class:`newton.solvers.experimental.coupled.SolverCoupledProxy`) and
  :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledAdmmSolverCfg`
  (linearized ADMM coupling, wrapping
  :class:`newton.solvers.experimental.coupled.SolverCoupledAdmm`). The manager
  partitions the Newton model into a source and destination entry, generically
  instantiates the sub-solvers from their cfg types, and dispatches on the
  config subclass to build the matching coupled solver.

* Added support for raw prim-path regex strings (e.g.
  ``"/World/envs/env_.*/MyCube"``) in the body-selector lists of
  :class:`~isaaclab_contrib.coupling.coupled_manager_cfg.CoupledSolverCfg`,
  alongside :class:`~isaaclab.managers.SceneEntityCfg` entries. Useful for
  claiming rigid assets that aren't registered as named scene entities.

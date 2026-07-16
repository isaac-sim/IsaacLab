Changed
^^^^^^^

* Removed the model-global ``shape_material_ke/kd/mu`` fields from
  :class:`~isaaclab_contrib.deformable.newton_manager_cfg.NewtonModelCfg`, which
  filled every rigid shape's material and clobbered per-asset materials. Set
  per-shape defaults through
  :class:`~isaaclab_newton.physics.NewtonShapeCfg` on ``NewtonCfg.default_shape_cfg``
  instead; per-asset materials now override those defaults. The model-global
  ``soft_contact_ke/kd/mu`` fields are unchanged.

Fixed
^^^^^

* Fixed proxy-coupled source solvers configured for external contacts to receive
  contacts from Newton's shared collision pipeline. Proxy destinations continue
  to use their entry-local collision pipeline.

Added
^^^^^

* Added :mod:`isaaclab_contrib.coupling`, exposing
  :class:`~isaaclab_contrib.coupling.coupler.NewtonCouplerManager`
  together with the
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerCfg`
  base config and two algorithm-specific subclasses:
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerProxyCfg`
  (wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledProxy`) and
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerAdmmCfg`
  (wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledADMM`).
  The coupler partitions the Newton model among explicit, named
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerEntryCfg`
  entries, instantiates each sub-solver from its config, and connects entries
  through named proxy mappings or symmetric ADMM contact pairs.

* Added support for prim-path regex strings (e.g.
  ``"/World/envs/env_.*/MyCube"``) in the body-selector lists of
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerEntryCfg`
  and :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerProxyMappingCfg`.
  Raw Newton body ids may also be given directly as integers in
  :attr:`~isaaclab_contrib.coupling.coupler_cfg.CouplerProxyMappingCfg.bodies`.

* Added :class:`~isaaclab_contrib.deformable.newton_manager_cfg.NewtonModelSolverCfg`,
  a shared solver-config base whose ``model_cfg``
  (:class:`~isaaclab_contrib.deformable.newton_manager_cfg.NewtonModelCfg`) is
  applied to the finalized Newton model. The VBD and coupler configs inherit it.

* Added implicit MPM support for coupled-solver entries, including per-entry
  substeps and in-place stepping.

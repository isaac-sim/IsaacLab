Added
^^^^^

* Added support for raw prim-path regex strings (e.g. ``"/World/envs/env_.*/MyCube"``)
  in the body-selector lists of
  :class:`~isaaclab_contrib.deformable.newton_manager_cfg.ProxyCoupledMJWarpVBDSolverCfg`,
  alongside :class:`~isaaclab.managers.SceneEntityCfg` entries. Useful for
  claiming rigid assets that aren't registered as named scene entities.

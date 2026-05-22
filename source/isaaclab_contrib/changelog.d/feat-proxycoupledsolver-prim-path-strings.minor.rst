Added
^^^^^

* Added :class:`~isaaclab_contrib.deformable.newton_manager_cfg.ProxyCoupledMJWarpVBDSolverCfg`
  and the matching
  :class:`~isaaclab_contrib.deformable.proxy_coupled_mjwarp_vbd_manager.NewtonProxyCoupledMJWarpVBDManager`,
  wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledProxy` to
  split simulation between MuJoCo Warp (rigids/articulations) and VBD
  (particles/deformables), with selected MJWarp bodies exposed as proxies in
  the VBD view.

* Added support for raw prim-path regex strings (e.g. ``"/World/envs/env_.*/MyCube"``)
  in the body-selector lists of
  :class:`~isaaclab_contrib.deformable.newton_manager_cfg.ProxyCoupledMJWarpVBDSolverCfg`,
  alongside :class:`~isaaclab.managers.SceneEntityCfg` entries. Useful for
  claiming rigid assets that aren't registered as named scene entities.

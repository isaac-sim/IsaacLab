Changed
^^^^^^^

* Removed ``viewer: ViewerCfg = ViewerCfg(...)`` from task environment configs following the
  removal of :class:`~isaaclab.envs.common.ViewerCfg`. Custom camera positions can be set via
  ``cfg.sim.visualizer_cfgs = [KitVisualizerCfg(eye=..., lookat=...)]``.

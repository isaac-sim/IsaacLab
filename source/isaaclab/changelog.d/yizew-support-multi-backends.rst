Added
^^^^^

* Added :attr:`~isaaclab.scene.InteractiveSceneCfg.class_type` so scene configs
  can instantiate custom scene classes.
* Added :meth:`~isaaclab.sim.SimulationContext.is_headless_or_exist_active_visualizer`
  to let kitless and external-visualizer demos share a visualizer-aware stepping
  condition.

Changed
^^^^^^^

* Updated demo scripts to support selectable PhysX or Newton MJWarp physics
  backends and Kit or Newton visualizers.

Added
^^^^^

* Added :meth:`~isaaclab.sim.SimulationContext.pre_render` to publish deferred physics state
  (e.g. Newton's Fabric/USD transform sync) to the rendering backend without a full
  :meth:`~isaaclab.sim.SimulationContext.render`, so a single consumer can refresh its transforms
  without stepping every other visualizer or firing every registered render callback.

Fixed
^^^^^

* Fixed on-demand visualizer recording to publish backend transforms before frame capture,
  preventing stale Newton poses in headless Kit videos. Capture now calls
  :meth:`~isaaclab.sim.SimulationContext.forward` and the new
  :meth:`~isaaclab.sim.SimulationContext.pre_render` directly instead of a full
  :meth:`~isaaclab.sim.SimulationContext.render`, avoiding double-rendering unrelated headless
  visualizers or invalidating another recorder's cached frame within the same physics step.
* Documented the render-state refresh required by custom on-demand visualizer recorders.
